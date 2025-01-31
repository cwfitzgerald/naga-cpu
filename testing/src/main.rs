fn main() {
    cc::Build::new()
        .out_dir("target/c")
        .target("x86_64-pc-windows-msvc")
        .opt_level(3)
        .std("c17")
        .flag_if_supported("-fno-strict-aliasing")
        .file("testing/c/test.c")
        .compile("test");
}
