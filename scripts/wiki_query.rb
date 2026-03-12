#!/usr/bin/env ruby
# frozen_string_literal: true

require 'json'
require 'net/http'
require 'optparse'
require 'pathname'
require 'sqlite3'
require 'uri'

DEFAULT_OLLAMA_URL = 'http://127.0.0.1:11434/api/embeddings'
DEFAULT_MODEL = 'embeddinggemma'
VEC_TABLE = 'wiki_vec'

def parse_args
  options = {
    db_path: 'data/wiki_simple.db',
    extension_path: nil,
    model: DEFAULT_MODEL,
    ollama_url: DEFAULT_OLLAMA_URL,
    k: 5,
    ef_search: 0,
    preview_chars: 260,
    vec_type: nil
  }

  parser = OptionParser.new do |opts|
    opts.banner = 'Usage: ruby scripts/wiki_query.rb [options] QUERY'
    opts.on('--db-path PATH', String, 'SQLite database path') { |v| options[:db_path] = v }
    opts.on('--extension-path PATH', String, 'Path to sqlite-vector extension') { |v| options[:extension_path] = v }
    opts.on('--model NAME', String, 'Ollama embedding model') { |v| options[:model] = v }
    opts.on('--ollama-url URL', String, 'Ollama embeddings endpoint') { |v| options[:ollama_url] = v }
    opts.on('--k N', Integer, 'Top K results') { |v| options[:k] = v }
    opts.on('--ef-search N', Integer, 'Runtime ef_search override') { |v| options[:ef_search] = v }
    opts.on('--preview-chars N', Integer, 'Preview length per chunk') { |v| options[:preview_chars] = v }
    opts.on('--vec-type TYPE', String, 'Override vector type (default: auto-detect from DB)') { |v| options[:vec_type] = v }
    opts.on('--[no-]quantize', '[DEPRECATED] Alias for --vec-type int8') { |v| options[:vec_type] = v ? 'int8' : 'float32' }
  end

  parser.parse!
  query = ARGV.join(' ').strip
  raise OptionParser::MissingArgument, 'QUERY' if query.empty?

  [options, query]
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

def embed_query(ollama_url, model, query)
  uri = URI(ollama_url)
  req = Net::HTTP::Post.new(uri)
  req['Content-Type'] = 'application/json'
  req.body = JSON.generate({ model: model, prompt: query })

  res = Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == 'https', read_timeout: 120) do |http|
    http.request(req)
  end

  raise "Ollama HTTP #{res.code}" unless res.code.to_i.between?(200, 299)

  body = JSON.parse(res.body)
  embedding = body['embedding']
  raise 'Unexpected Ollama response' unless embedding.is_a?(Array) && !embedding.empty?

  embedding.map(&:to_f)
end

def truncate_text(text, width)
  normalized = text.to_s.gsub(/\s+/, ' ').strip
  return normalized if normalized.length <= width

  "#{normalized[0...[width - 3, 0].max].rstrip}..."
end

def db_stats(conn)
  total_articles = conn.get_first_value('SELECT COUNT(*) FROM wiki_articles').to_i
  total_chunks = conn.get_first_value('SELECT COUNT(*) FROM wiki_chunks').to_i
  { articles: total_articles, chunks: total_chunks }
end

def run_query(conn, query_embedding, k, ef_search)
  vector_text = JSON.generate(query_embedding)
  sql = <<~SQL
    SELECT a.title, c.chunk_index, c.content, v.distance
    FROM #{VEC_TABLE} AS v
    JOIN wiki_chunks AS c ON c.id = v.rowid
    JOIN wiki_articles AS a ON a.id = c.article_id
    WHERE #{VEC_TABLE} MATCH ?
  SQL

  params = [vector_text]
  if ef_search.positive?
    sql += ' AND ef_search = ?'
    params << ef_search
  end

  sql += ' LIMIT ?'
  params << k

  stmt = conn.prepare(sql)
  rows = []
  begin
    stmt.execute(*params) do |result|
      result.each_hash { |row| rows << row }
    end
  ensure
    stmt.close
  end
  rows
end

def main
  options, query = parse_args
  repo_root = Pathname.new(File.expand_path('..', __dir__))
  extension_path = options[:extension_path] ? Pathname.new(options[:extension_path]) : detect_extension_path(repo_root)

  conn = nil
  conn = SQLite3::Database.new(options[:db_path])
  conn.results_as_hash = true
  conn.enable_load_extension(true)
  conn.load_extension(extension_path.to_s)

  vec_type = if options[:vec_type].nil?
               detected = conn.get_first_value("SELECT value FROM #{VEC_TABLE}_config WHERE key = 'type'")
               detected || 'float32'
             else
               options[:vec_type]
             end

  started_at = Time.now

  stats = db_stats(conn)

  embed_started = Time.now
  query_embedding = embed_query(options[:ollama_url], options[:model], query)
  embed_elapsed = Time.now - embed_started

  search_started = Time.now
  rows = run_query(conn, query_embedding, options[:k], options[:ef_search])
  search_elapsed = Time.now - search_started

  total_elapsed = Time.now - started_at

  puts format('[db]    articles=%d  chunks=%d  type=%s', stats[:articles], stats[:chunks], vec_type)
  puts format('[time]  embed=%.3fs  search=%.3fs  total=%.3fs', embed_elapsed, search_elapsed, total_elapsed)
  puts format('[query] %s  results=%d/%d requested', query, rows.length, options[:k])

  if rows.empty?
    puts 'No results found.'
    return 0
  end

  puts

  rows.each_with_index do |row, idx|
    title = row['title']
    chunk_index = row['chunk_index']
    distance = row['distance'].to_f
    similarity = 1.0 - distance
    preview = truncate_text(row['content'], options[:preview_chars])

    puts format('%d. %s (chunk %d, distance=%.4f, similarity=%.1f%%)', idx + 1, title, chunk_index, distance,
                similarity * 100)
    puts "   #{preview}"
    puts
  end

  0
ensure
  conn&.close
end

exit(main)
