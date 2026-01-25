#[cfg(test)]
mod tests {
    use crate::models::binary_reader::BinaryReader;
    use std::fs::File;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_u8() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&[42u8]).unwrap();
        file.flush().unwrap();

        let mut f = File::open(file.path()).unwrap();
        let value = BinaryReader::read_u8(&mut f).unwrap();
        assert_eq!(value, 42);
    }

    #[test]
    fn test_read_u32() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&42u32.to_le_bytes()).unwrap();
        file.flush().unwrap();

        let mut f = File::open(file.path()).unwrap();
        let value = BinaryReader::read_u32(&mut f).unwrap();
        assert_eq!(value, 42);
    }

    #[test]
    fn test_read_u64() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&9876543210u64.to_le_bytes()).unwrap();
        file.flush().unwrap();

        let mut f = File::open(file.path()).unwrap();
        let value = BinaryReader::read_u64(&mut f).unwrap();
        assert_eq!(value, 9876543210);
    }

    #[test]
    fn test_read_string() {
        let mut file = NamedTempFile::new().unwrap();
        let test_str = "hello";
        file.write_all(&(test_str.len() as u32).to_le_bytes())
            .unwrap();
        file.write_all(test_str.as_bytes()).unwrap();
        file.flush().unwrap();

        let mut f = File::open(file.path()).unwrap();
        let value = BinaryReader::read_string(&mut f).unwrap();
        assert_eq!(value, "hello");
    }
}
