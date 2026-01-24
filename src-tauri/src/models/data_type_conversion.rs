use super::gguf_data_type::GGUFDataType;
use std::fmt;

impl GGUFDataType {
    /// Convert type number to enum
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(GGUFDataType::F32),
            1 => Some(GGUFDataType::F16),
            2 => Some(GGUFDataType::Q8_0),
            3 => Some(GGUFDataType::Q8_1),
            4 => Some(GGUFDataType::Q4_0),
            5 => Some(GGUFDataType::Q4_1),
            6 => Some(GGUFDataType::Q5_0),
            7 => Some(GGUFDataType::Q5_1),
            8 => Some(GGUFDataType::Q2_K),
            9 => Some(GGUFDataType::Q3_K),
            10 => Some(GGUFDataType::Q4_K),
            11 => Some(GGUFDataType::Q5_K),
            12 => Some(GGUFDataType::Q6_K),
            13 => Some(GGUFDataType::I8),
            14 => Some(GGUFDataType::I16),
            15 => Some(GGUFDataType::I32),
            _ => None,
        }
    }
}

impl fmt::Display for GGUFDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GGUFDataType::F32 => write!(f, "F32"),
            GGUFDataType::F16 => write!(f, "F16"),
            GGUFDataType::Q8_0 => write!(f, "Q8_0"),
            GGUFDataType::Q8_1 => write!(f, "Q8_1"),
            GGUFDataType::Q4_0 => write!(f, "Q4_0"),
            GGUFDataType::Q4_1 => write!(f, "Q4_1"),
            GGUFDataType::Q5_0 => write!(f, "Q5_0"),
            GGUFDataType::Q5_1 => write!(f, "Q5_1"),
            GGUFDataType::Q2_K => write!(f, "Q2_K"),
            GGUFDataType::Q3_K => write!(f, "Q3_K"),
            GGUFDataType::Q4_K => write!(f, "Q4_K"),
            GGUFDataType::Q5_K => write!(f, "Q5_K"),
            GGUFDataType::Q6_K => write!(f, "Q6_K"),
            GGUFDataType::I8 => write!(f, "I8"),
            GGUFDataType::I16 => write!(f, "I16"),
            GGUFDataType::I32 => write!(f, "I32"),
            GGUFDataType::Count => write!(f, "Count"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_conversion() {
        assert_eq!(GGUFDataType::from_u32(0), Some(GGUFDataType::F32));
        assert_eq!(GGUFDataType::from_u32(1), Some(GGUFDataType::F16));
        assert_eq!(GGUFDataType::from_u32(2), Some(GGUFDataType::Q8_0));
        assert_eq!(GGUFDataType::from_u32(255), None);
    }

    #[test]
    fn test_data_type_display() {
        assert_eq!(GGUFDataType::F32.to_string(), "F32");
        assert_eq!(GGUFDataType::Q4_0.to_string(), "Q4_0");
        assert_eq!(GGUFDataType::I32.to_string(), "I32");
    }
}
