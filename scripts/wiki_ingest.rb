#!/usr/bin/env ruby
# frozen_string_literal: true

require 'English'
require 'fileutils'
require 'json'
require 'loofah'
require 'net/http'
require 'nokogiri'
require 'optparse'
require 'pathname'
require 'sqlite3'
require 'uri'

DEFAULT_DUMP_URL = 'https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2'
DEFAULT_OLLAMA_URL = 'http://127.0.0.1:11434/api/embeddings'
DEFAULT_MODEL = 'embeddinggemma'
VEC_TABLE = 'wiki_vec'

def parse_args
  options = {
    db_path: 'data/wiki_simple.db',
    extension_path: nil,
    dump_url: DEFAULT_DUMP_URL,
    dump_path: 'data/simplewiki-latest-pages-articles.xml.bz2',
    model: DEFAULT_MODEL,
    ollama_url: DEFAULT_OLLAMA_URL,
    max_articles: 10_000,
    chunk_max_chars: 1200,
    min_chunk_chars: 200,
    batch_size: 500,
    concurrency: 4,
    maintenance_every_chunks: 0,
    checkpoint_mode: 'PASSIVE',
    post_ingest_optimize: true,
    force_redownload: false,
    vec_type: 'float32',
    hnsw_m: 16,
    ef_construction: 200
  }

  OptionParser.new do |opts|
    opts.banner = 'Usage: ruby scripts/wiki_ingest.rb [options]'
    opts.on('--db-path PATH', String, 'SQLite database path') { |v| options[:db_path] = v }
    opts.on('--extension-path PATH', String, 'Path to sqlite-vector extension') { |v| options[:extension_path] = v }
    opts.on('--dump-url URL', String, 'Simple Wikipedia dump URL') { |v| options[:dump_url] = v }
    opts.on('--dump-path PATH', String, 'Local dump path') { |v| options[:dump_path] = v }
    opts.on('--model NAME', String, 'Ollama embedding model') { |v| options[:model] = v }
    opts.on('--ollama-url URL', String, 'Ollama embeddings endpoint') { |v| options[:ollama_url] = v }
    opts.on('--max-articles N', Integer, 'Max articles (0 means all)') { |v| options[:max_articles] = v }
    opts.on('--chunk-max-chars N', Integer, 'Max chars per chunk') { |v| options[:chunk_max_chars] = v }
    opts.on('--min-chunk-chars N', Integer, 'Min chars for merged chunks') { |v| options[:min_chunk_chars] = v }
    opts.on('--batch-size N', Integer, 'Commit batch size by chunks') { |v| options[:batch_size] = v }
    opts.on('--concurrency N', Integer, 'Concurrent Ollama requests (default: 4)') { |v| options[:concurrency] = v }
    opts.on('--maintenance-every-chunks N', Integer, 'Run maintenance every N inserted chunks (0 disables)') { |v| options[:maintenance_every_chunks] = v }
    opts.on('--checkpoint-mode MODE', String, 'WAL checkpoint mode: PASSIVE|FULL|RESTART|TRUNCATE') { |v| options[:checkpoint_mode] = v }
    opts.on('--[no-]post-ingest-optimize', 'Run PRAGMA optimize after ingest (default: true)') { |v| options[:post_ingest_optimize] = v }
    opts.on('--force-redownload', 'Redownload dump from scratch') { options[:force_redownload] = true }
    opts.on('--vec-type TYPE', String, 'Vector storage type: float32, int8, binary (default: float32)') { |v| options[:vec_type] = v }
    opts.on('--[no-]quantize', '[DEPRECATED] Alias for --vec-type int8') { |v| options[:vec_type] = v ? 'int8' : 'float32' }
    opts.on('--hnsw-m N', Integer, 'HNSW connections per layer (default: 16)') { |v| options[:hnsw_m] = v }
    opts.on('--ef-construction N', Integer, 'HNSW build-time beam width (default: 200)') { |v| options[:ef_construction] = v }
  end.parse!

  options
end

def normalize_checkpoint_mode(raw)
  mode = raw.to_s.upcase
  allowed = %w[PASSIVE FULL RESTART TRUNCATE]
  raise "Invalid --checkpoint-mode '#{raw}'. Use one of: #{allowed.join(', ')}" unless allowed.include?(mode)

  mode
end

def run_maintenance(conn, checkpoint_mode:, reason:)
  started = Time.now
  begin
    conn.execute("PRAGMA wal_checkpoint(#{checkpoint_mode})")
    puts "[maint] #{reason} checkpoint=#{checkpoint_mode}"
  rescue SQLite3::LockedException
    puts "[maint] #{reason} checkpoint skipped (table locked)"
  end
  puts format('[maint] %s total=%.2fs', reason, Time.now - started)
end

def detect_extension_path(repo_root)
  patterns = [
    repo_root.join('build/*/*/release/libsqlite_vector.dylib').to_s,
    repo_root.join('build/*/*/release/libsqlite_vector.so').to_s,
    repo_root.join('build/*/*/release/libsqlite_vector.dll').to_s
  ]

  patterns.each do |pattern|
    matches = Dir.glob(pattern).sort
    return Pathname.new(matches.first) unless matches.empty?
  end

  raise 'Could not auto-detect sqlite-vector extension in build/*/*/release'
end

def download_dump(url, target, force_redownload: false)
  target = Pathname.new(target)
  FileUtils.mkdir_p(target.dirname)
  File.delete(target) if force_redownload && target.exist?

  existing_size = target.exist? ? target.size : 0
  uri = URI(url)
  req = Net::HTTP::Get.new(uri)
  req['Range'] = "bytes=#{existing_size}-" if existing_size.positive?

  mode = existing_size.positive? ? 'ab' : 'wb'

  Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == 'https') do |http|
    http.request(req) do |res|
      if res.code.to_i == 416
        puts "[download] already complete: #{target}"
        return
      end

      if res.code.to_i == 200 && existing_size.positive?
        mode = 'wb'
        existing_size = 0
      end

      raise "Download failed (HTTP #{res.code})" unless res.code.to_i.between?(200, 299)

      total = res['Content-Length']&.to_i
      total_bytes = total ? total + existing_size : nil
      written = existing_size

      File.open(target, mode) do |file|
        res.read_body do |chunk|
          file.write(chunk)
          written += chunk.bytesize
          next unless total_bytes

          pct = (written * 100.0 / total_bytes)
          print format("\r[download] %.1f MiB (%.1f%%)", written / (1024.0 * 1024.0), pct)
          $stdout.flush
        end
      end

      puts if total_bytes
    end
  end

  puts "[download] saved to #{target}"
end

def first_text(node, xpath)
  found = node.at_xpath(xpath)
  found&.text
end

def iter_wiki_pages(xml_bz2_path)
  Enumerator.new do |yielder|
    cmd = ['bzcat', xml_bz2_path.to_s]
    IO.popen(cmd, 'r') do |io|
      Nokogiri::XML::Reader(io).each do |reader|
        next unless reader.node_type == Nokogiri::XML::Reader::TYPE_ELEMENT
        next unless reader.name == 'page'

        page_doc = Nokogiri::XML(reader.outer_xml)
        page = page_doc.at_xpath("/*[local-name()='page']")
        next unless page

        ns = first_text(page, "./*[local-name()='ns']")
        next unless ns == '0'

        next if page.at_xpath("./*[local-name()='redirect']")

        page_id_text = first_text(page, "./*[local-name()='id']")
        title = first_text(page, "./*[local-name()='title']")
        text = first_text(page, "./*[local-name()='revision']/*[local-name()='text']")
        next if page_id_text.nil? || title.nil? || text.nil? || text.strip.empty?

        begin
          page_id = Integer(page_id_text)
        rescue ArgumentError
          next
        end

        yielder << [page_id, title, text]
      end
    end

    raise 'bzcat failed. Ensure bzip2 tools are installed.' if $CHILD_STATUS&.exitstatus.to_i != 0
  end
end

def strip_wiki_markup(text)
  # Pre-sanitize: strip HTML tags (ref, gallery, nowiki, etc.) that confuse Pandoc
  pre = Loofah.fragment(text).scrub!(:prune).to_text

  plain = IO.popen(['pandoc', '-f', 'mediawiki', '-t', 'plain', '--wrap=none'], 'r+', err: '/dev/null') do |io|
    io.write(pre)
    io.close_write
    io.read
  end

  unless $CHILD_STATUS&.success? && !plain.nil? && !plain.strip.empty?
    # Fallback for malformed markup that Pandoc rejects
    plain = pre.dup
    plain.gsub!(/\{\|.*?\|\}/m, ' ')
    plain.gsub!(/\{\{[^{}]*\}\}/m, ' ')
    plain.gsub!(/\[\[(?:File|Image|Category):[^\]]+\]\]/i, ' ')
    plain.gsub!(/\[\[([^\]|]+)\|([^\]]+)\]\]/, '\\2')
    plain.gsub!(/\[\[([^\]]+)\]\]/, '\\1')
    plain.gsub!(/\[https?:\/\/\S+\s+([^\]]+)\]/, '\\1')
    plain.gsub!(/\[https?:\/\/\S+\]/, ' ')
    plain.gsub!(/^=+\s*(.*?)\s*=+$/, '\\1')
    plain.gsub!(/''+/, '')
  end

  # Post-sanitize: strip any residual HTML from Pandoc output
  cleaned = Loofah.fragment(plain).scrub!(:strip).to_text
  cleaned.gsub!(/\n{3,}/, "\n\n")
  cleaned.strip
end

def split_paragraph_chunks(text, max_chars, min_chunk_chars)
  raw_paragraphs = text.split(/\n\s*\n/).map(&:strip).reject(&:empty?)
  paragraphs = []

  raw_paragraphs.each do |para|
    normalized = para.gsub(/\s+/, ' ').strip
    next if normalized.empty?

    while normalized.length > max_chars
      split_at = normalized.rindex(' ', max_chars) || max_chars
      split_at = max_chars if split_at < (max_chars / 2)
      paragraphs << normalized[0...split_at].strip
      normalized = normalized[split_at..].to_s.strip
    end

    paragraphs << normalized unless normalized.empty?
  end

  chunks = []
  current = []
  current_len = 0

  paragraphs.each do |para|
    extra = para.length + (current.empty? ? 0 : 2)
    if !current.empty? && (current_len + extra > max_chars)
      chunk = current.join("\n\n").strip
      chunks << chunk unless chunk.empty?
      current = [para]
      current_len = para.length
    else
      current << para
      current_len += extra
    end
  end

  unless current.empty?
    chunk = current.join("\n\n").strip
    chunks << chunk unless chunk.empty?
  end

  if chunks.length > 1 && chunks[-1].length < min_chunk_chars
    chunks[-2] = [chunks[-2], chunks[-1]].join("\n\n").strip
    chunks.pop
  end

  chunks.reject { |c| c.empty? || c.length < 20 }
end

def embed_text(http, uri, model, text, retries: 3)
  payload = JSON.generate({ model: model, prompt: text })

  1.upto(retries) do |attempt|
    req = Net::HTTP::Post.new(uri)
    req['Content-Type'] = 'application/json'
    req.body = payload

    res = http.request(req)

    raise "Ollama HTTP #{res.code}" unless res.code.to_i.between?(200, 299)

    body = JSON.parse(res.body)
    embedding = body['embedding']
    raise 'Unexpected Ollama response' unless embedding.is_a?(Array) && !embedding.empty?

    return embedding.map(&:to_f)
  rescue StandardError
    raise if attempt == retries

    sleep(attempt * 1.5)
  end

  raise 'unreachable'
end

def embed_batch(ollama_url, model, texts, concurrency)
  uri = URI(ollama_url)
  results = Array.new(texts.length)
  queue = Queue.new
  texts.each_with_index { |t, i| queue << [i, t] }
  concurrency.times { queue << nil }

  threads = Array.new(concurrency) do
    Thread.new do
      Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == 'https', read_timeout: 120) do |http|
        while (item = queue.pop)
          idx, text = item
          results[idx] = embed_text(http, uri, model, text)
        end
      end
    end
  end

  threads.each(&:join)
  results
end

def configure_for_bulk_ingest(conn)
  conn.execute_batch(<<~SQL)
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    PRAGMA cache_size = -64000;
    PRAGMA mmap_size = 268435456;
    PRAGMA temp_store = MEMORY;
    PRAGMA foreign_keys = ON;
  SQL
end

def ensure_schema(conn)
  conn.execute_batch(<<~SQL);
    CREATE TABLE IF NOT EXISTS wiki_articles (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      source_page_id INTEGER NOT NULL UNIQUE,
      title TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS wiki_chunks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      article_id INTEGER NOT NULL,
      chunk_index INTEGER NOT NULL,
      content TEXT NOT NULL,
      FOREIGN KEY(article_id) REFERENCES wiki_articles(id) ON DELETE CASCADE,
      UNIQUE(article_id, chunk_index)
    );

    CREATE TABLE IF NOT EXISTS ingest_state (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_wiki_chunks_article_id ON wiki_chunks(article_id);
    CREATE INDEX IF NOT EXISTS idx_wiki_articles_title ON wiki_articles(title);
  SQL
end

def get_state(conn, key)
  row = conn.get_first_row('SELECT value FROM ingest_state WHERE key = ?', key)
  row&.first
end

def set_state(conn, stmts, key, value)
  if stmts
    stmts[:set_state].execute(key, value)
  else
    conn.execute(
      'INSERT INTO ingest_state(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value',
      [key, value]
    )
  end
end

def ensure_vector_table(conn, dims, m:, ef_construction:, vec_type:)
  current_dims = get_state(conn, 'vec_dims')
  if current_dims && current_dims.to_i != dims
    raise "Existing vector dims=#{current_dims}, but new embeddings use dims=#{dims}"
  end

  exists = conn.get_first_row(
    "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
    VEC_TABLE
  )

  unless exists
    conn.execute(
      "CREATE VIRTUAL TABLE #{VEC_TABLE} USING vec0(dims=#{dims}, metric=cosine, ef_search=50, m=#{m}, ef_construction=#{ef_construction}, type=#{vec_type})"
    )
  end

  set_state(conn, nil, 'vec_dims', dims.to_s)
  set_state(conn, nil, 'vec_type', vec_type)
end

def upsert_article(conn, stmts, page_id, title)
  stmts[:upsert_article].execute(page_id, title)
  rs = stmts[:lookup_article].execute(page_id)
  row = rs.next
  raise "Failed to resolve article id for page_id=#{page_id}" if row.nil?

  row[0].to_i
end

def prepare_ingest_stmts(conn)
  {
    ins_chunk: conn.prepare('INSERT OR IGNORE INTO wiki_chunks(article_id, chunk_index, content) VALUES (?, ?, ?)'),
    ins_vec: conn.prepare("INSERT INTO #{VEC_TABLE}(rowid, vector) VALUES (?, vec(?))"),
    upsert_article: conn.prepare(
      'INSERT INTO wiki_articles(source_page_id, title) VALUES (?, ?) ON CONFLICT(source_page_id) DO UPDATE SET title = excluded.title'
    ),
    lookup_article: conn.prepare('SELECT id FROM wiki_articles WHERE source_page_id = ?'),
    set_state: conn.prepare(
      'INSERT INTO ingest_state(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value'
    )
  }
end

def finalize_stmts(stmts)
  stmts.each_value(&:close)
end

def insert_chunk_and_vector(conn, stmts, article_id, chunk_index, content, embedding)
  stmts[:ins_chunk].execute(article_id, chunk_index, content)
  return false if conn.changes.zero?

  chunk_id = conn.last_insert_row_id
  vector_text = JSON.generate(embedding)
  stmts[:ins_vec].execute(chunk_id, vector_text)
  true
end

def begin_tx(conn)
  conn.execute('BEGIN IMMEDIATE')
end

def commit_tx(conn)
  conn.execute('COMMIT')
end

def main
  options = parse_args
  options[:checkpoint_mode] = normalize_checkpoint_mode(options[:checkpoint_mode])
  repo_root = Pathname.new(File.expand_path('..', __dir__))
  extension_path = options[:extension_path] ? Pathname.new(options[:extension_path]) : detect_extension_path(repo_root)
  dump_path = Pathname.new(options[:dump_path])
  db_path = Pathname.new(options[:db_path])

  puts "[setup] extension: #{extension_path}"
  puts "[setup] database: #{db_path}"
  puts "[setup] dump: #{dump_path}"

  download_dump(options[:dump_url], dump_path, force_redownload: options[:force_redownload])

  FileUtils.mkdir_p(db_path.dirname)
  conn = SQLite3::Database.new(db_path.to_s)
  conn.enable_load_extension(true)
  conn.load_extension(extension_path.to_s)
  configure_for_bulk_ingest(conn)
  ensure_schema(conn)

  last_page_id = (get_state(conn, 'last_page_id') || '0').to_i
  puts "[resume] last_page_id=#{last_page_id}"

  processed_articles = 0
  stored_chunks = 0
  skipped_articles = 0
  total_inserted_chunks = 0
  last_maintenance_at = 0
  vec_dims_state = get_state(conn, 'vec_dims')
  vector_dims = vec_dims_state&.to_i
  embedding_seconds = 0.0
  started_at = Time.now

  tx_open = false
  stmts = nil
  buffer = [] # [{page_id:, title:, chunk_index:, content:}]

  flush_buffer = lambda do
    return if buffer.empty?

    embed_started = Time.now
    embeddings = embed_batch(options[:ollama_url], options[:model], buffer.map { |b| b[:content] }, options[:concurrency])
    embedding_seconds += (Time.now - embed_started)

    if vector_dims.nil?
      vector_dims = embeddings[0].length
      ensure_vector_table(conn, vector_dims, m: options[:hnsw_m], ef_construction: options[:ef_construction], vec_type: options[:vec_type])
      stmts = prepare_ingest_stmts(conn)
      puts "[setup] created #{VEC_TABLE} with dims=#{vector_dims} m=#{options[:hnsw_m]} ef_construction=#{options[:ef_construction]} type=#{options[:vec_type]}"
    end
    stmts ||= prepare_ingest_stmts(conn)

    begin_tx(conn) unless tx_open
    tx_open = true

    last_pid = nil
    current_article_id = nil
    buffer.each_with_index do |item, i|
      emb = embeddings[i]
      raise "Embedding dimension changed from #{vector_dims} to #{emb.length}" if emb.length != vector_dims

      if item[:page_id] != last_pid
        current_article_id = upsert_article(conn, stmts, item[:page_id], item[:title])
        last_pid = item[:page_id]
      end

      if insert_chunk_and_vector(conn, stmts, current_article_id, item[:chunk_index], item[:content], emb)
        total_inserted_chunks += 1
      end
    end

    commit_tx(conn)
    tx_open = false

    set_state(conn, stmts, 'last_page_id', buffer.last[:page_id].to_s)

    if options[:maintenance_every_chunks].positive? &&
       (total_inserted_chunks - last_maintenance_at) >= options[:maintenance_every_chunks]
      run_maintenance(conn, checkpoint_mode: options[:checkpoint_mode], reason: "periodic chunks=#{total_inserted_chunks}")
      last_maintenance_at = total_inserted_chunks
    end

    buffer.clear
  end

  begin
    iter_wiki_pages(dump_path).each do |page_id, title, raw_text|
      next if page_id <= last_page_id

      cleaned = strip_wiki_markup(raw_text)
      chunks = split_paragraph_chunks(cleaned, options[:chunk_max_chars], options[:min_chunk_chars])

      if chunks.empty?
        skipped_articles += 1
        next
      end

      chunks.each_with_index do |chunk, idx|
        buffer << { page_id: page_id, title: title, chunk_index: idx, content: chunk }
      end

      processed_articles += 1
      stored_chunks += chunks.length
      last_page_id = page_id

      flush_buffer.call if buffer.length >= options[:batch_size]

      if (processed_articles % 25).zero?
        elapsed = [Time.now - started_at, 1e-6].max
        rate = stored_chunks / elapsed
        puts format(
          '[progress] articles=%d chunks=%d skipped=%d rate=%.2f chunks/s embed=%.1fs total=%.1fs',
          processed_articles,
          stored_chunks,
          skipped_articles,
          rate,
          embedding_seconds,
          elapsed
        )
      end

      if options[:max_articles].positive? && processed_articles >= options[:max_articles]
        puts "[stop] reached --max-articles=#{options[:max_articles]}"
        break
      end
    end

    flush_buffer.call

    if options[:post_ingest_optimize]
      finalize_stmts(stmts) if stmts
      stmts = nil
      conn.execute('PRAGMA optimize')
      puts '[post-ingest] PRAGMA optimize done'
      run_maintenance(conn, checkpoint_mode: options[:checkpoint_mode], reason: 'post-ingest')
    end
  rescue Interrupt
    puts "\n[interrupt] flushing buffer and committing before exit"
    flush_buffer.call rescue nil
  ensure
    finalize_stmts(stmts) if stmts
    conn.close
  end

  elapsed = [Time.now - started_at, 1e-6].max
  puts format(
    '[done] processed_articles=%d stored_chunks=%d skipped_articles=%d embed=%.1fs total=%.1fs',
    processed_articles,
    stored_chunks,
    skipped_articles,
    embedding_seconds,
    elapsed
  )

  0
end

exit(main)
