use camino::Utf8Path;

fn main() {
    let files_in_snapshot = Utf8Path::new("snapshots/in")
        .read_dir_utf8()
        .unwrap()
        .into_iter()
        .filter_map(|entry| entry.ok());

    for file in files_in_snapshot {
        if !file.file_type().unwrap().is_file() {
            continue;
        }

        let file_path = file.path();

        println!("Translating {file_path}");

        let contents = std::fs::read_to_string(&file_path).unwrap();

        let module = naga::front::wgsl::parse_str(&contents).unwrap();

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );

        let module_info = validator.validate(&module).unwrap();
    }
}
