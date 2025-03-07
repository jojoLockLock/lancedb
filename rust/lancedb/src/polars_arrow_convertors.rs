#[cfg(test)]
mod test_polars_arrow_convertors {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::DataType as ArrowDataType;
    use arrow::record_batch::RecordBatch;
    use polars::prelude::{DataType as PolarsDType, Field as PolarField, Schema as PolarSchema};

    #[test]
    fn test_convert_polars_df_schema_to_arrow_rb_schema() {
        let mut polars_schema = PolarSchema::new();
        polars_schema.with_column("id".into(), PolarsDType::Int32);
        polars_schema.with_column("name".into(), PolarsDType::Utf8);

        let arrow_schema = convert_polars_df_schema_to_arrow_rb_schema(polars_schema).unwrap();

        assert_eq!(arrow_schema.fields().len(), 2);
        assert_eq!(arrow_schema.field(0).name(), "id");
        assert_eq!(arrow_schema.field(0).data_type(), &ArrowDataType::Int32);
        assert_eq!(arrow_schema.field(1).name(), "name");
        assert_eq!(arrow_schema.field(1).data_type(), &ArrowDataType::Utf8);
    }

    #[test]
    fn test_convert_arrow_rb_schema_to_polars_df_schema() {
        let arrow_fields = vec![
            arrow_schema::Field::new("id", ArrowDataType::Int32, true),
            arrow_schema::Field::new("name", ArrowDataType::Utf8, true),
        ];
        let arrow_schema = arrow_schema::Schema::new(arrow_fields);

        let polars_schema = convert_arrow_rb_schema_to_polars_df_schema(&arrow_schema).unwrap();

        assert_eq!(polars_schema.len(), 2);
        assert_eq!(polars_schema.get_field_by_name("id").unwrap().data_type(), &PolarsDType::Int32);
        assert_eq!(polars_schema.get_field_by_name("name").unwrap().data_type(), &PolarsDType::Utf8);
    }

    #[test]
    fn test_convert_arrow_rb_to_polars_df() {
        let id_array = Int32Array::from(vec![1, 2, 3]);
        let name_array = StringArray::from(vec!["a", "b", "c"]);

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", ArrowDataType::Int32, true),
            arrow_schema::Field::new("name", ArrowDataType::Utf8, true),
        ]);

        let record_batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(id_array), Arc::new(name_array)],
        ).unwrap();

        let mut polars_schema = PolarSchema::new();
        polars_schema.with_column("id".into(), PolarsDType::Int32);
        polars_schema.with_column("name".into(), PolarsDType::Utf8);

        let df = convert_arrow_rb_to_polars_df(&record_batch, &polars_schema).unwrap();

        assert_eq!(df.shape(), (3, 2));
        assert_eq!(df.schema().len(), 2);
    }

    #[test]
    fn test_convert_polars_arrow_array_to_arrow_rs_array() {
        let polars_array = polars_arrow::array::Int32Array::from_slice(&[1, 2, 3]);
        let arrow_array = convert_polars_arrow_array_to_arrow_rs_array(
            Box::new(polars_array),
            ArrowDataType::Int32,
        ).unwrap();

        assert_eq!(arrow_array.len(), 3);
        assert_eq!(arrow_array.data_type(), &ArrowDataType::Int32);
    }

    #[test]
    fn test_convert_arrow_rs_array_to_polars_arrow_array() {
        let arrow_array = Arc::new(Int32Array::from(vec![1, 2, 3])) as Arc<dyn arrow_array::Array>;
        let polars_array = convert_arrow_rs_array_to_polars_arrow_array(
            &arrow_array,
            polars::datatypes::ArrowDataType::Int32,
        ).unwrap();

        assert_eq!(polars_array.len(), 3);
        assert_eq!(polars_array.data_type(), &polars::datatypes::ArrowDataType::Int32);
    }

    #[test]
    fn test_convert_polars_arrow_field_to_arrow_rs_field() {
        let polars_field = polars_arrow::datatypes::Field::new(
            "test",
            polars::datatypes::ArrowDataType::Int32,
            true,
        );

        let arrow_field = convert_polars_arrow_field_to_arrow_rs_field(polars_field).unwrap();

        assert_eq!(arrow_field.name(), "test");
        assert_eq!(arrow_field.data_type(), &ArrowDataType::Int32);
        assert!(arrow_field.is_nullable());
    }

    #[test]
    fn test_convert_arrow_rs_field_to_polars_arrow_field() {
        let arrow_field = arrow_schema::Field::new("test", ArrowDataType::Int32, true);

        let polars_field = convert_arrow_rs_field_to_polars_arrow_field(&arrow_field).unwrap();

        assert_eq!(polars_field.name, "test");
        assert_eq!(polars_field.data_type(), &polars::datatypes::ArrowDataType::Int32);
        assert!(polars_field.is_nullable);
    }
}
