set_project("sqlite-vector")
set_version("0.1.0")
set_languages("c11")

add_rules("mode.debug", "mode.release")

if is_mode("debug") then
    add_defines("DEBUG")
    set_symbols("debug")
    set_optimize("none")
else
    set_optimize("fastest")
    set_strip("all")
end

target("sqlite_vector")
set_kind("shared")

add_files("src/*.c")

add_includedirs("include")
add_includedirs("third_party/simsimd/include")

-- SQLite extension symbols are provided by the host process at load time
if is_plat("macosx") then
    add_ldflags("-undefined dynamic_lookup", {
        force = true
    })
    add_shflags("-undefined dynamic_lookup", {
        force = true
    })
end

if is_plat("linux") then
    add_cflags("-fPIC")
    add_links("m") -- sqrt etc. from libm
end
