#[cfg(test)]
mod test_polars_arrow_conversions {
    use super::*;
    use arrow::datatypes::DataType as ArrowDataType;
    use arrow_array::{Int32Array, StringArray};
    use polars::prelude::*;
    use std::sync::Arc;

    #[test]
    fn test_convert_polars_df_schema_to_arrow_rb_schema() -> Result<()> {
        let mut polars_schema = Schema::new();
        polars_schema.with_column("int_col".to_string(), DataType::Int32);
        polars_schema.with_column("str_col".to_string(), DataType::Utf8);

        let arrow_schema = convert_polars_df_schema_to_arrow_rb_schema(polars_schema)?;

        assert_eq!(arrow_schema.fields().len(), 2);
        assert_eq!(arrow_schema.field(0).name(), "int_col");
        assert_eq!(arrow_schema.field(0).data_type(), &ArrowDataType::Int32);
        assert_eq!(arrow_schema.field(1).name(), "str_col");
        assert_eq!(arrow_schema.field(1).data_type(), &ArrowDataType::Utf8);
        Ok(())
    }

    #[test]
    fn test_convert_arrow_rb_schema_to_polars_df_schema() -> Result<()> {
        let arrow_fields = vec![
            arrow_schema::Field::new("int_col", ArrowDataType::Int32, true),
            arrow_schema::Field::new("str_col", ArrowDataType::Utf8, true),
        ];
        let arrow_schema = arrow_schema::Schema::new(arrow_fields);

        let polars_schema = convert_arrow_rb_schema_to_polars_df_schema(&arrow_schema)?;

        assert_eq!(polars_schema.len(), 2);
        assert_eq!(polars_schema.get_field_by_name("int_col").unwrap().data_type(), &DataType::Int32);
        assert_eq!(polars_schema.get_field_by_name("str_col").unwrap().data_type(), &DataType::Utf8);
        Ok(())
    }

    #[test]
    fn test_convert_arrow_rb_to_polars_df() -> Result<()> {
        let int_array = Int32Array::from(vec![1, 2, 3]);
        let str_array = StringArray::from(vec!["a", "b", "c"]);

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("int_col", ArrowDataType::Int32, true),
            arrow_schema::Field::new("str_col", ArrowDataType::Utf8, true),
        ]);

        let record_batch = arrow::record_batch::RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(int_array), Arc::new(str_array)],
        )?;

        let mut polars_schema = Schema::new();
        polars_schema.with_column("int_col".to_string(), DataType::Int32);
        polars_schema.with_column("str_col".to_string(), DataType::Utf8);

        let df = convert_arrow_rb_to_polars_df(&record_batch, &polars_schema)?;

        assert_eq!(df.shape(), (3, 2));
        assert_eq!(df.column("int_col")?.dtype(), &DataType::Int32);
        assert_eq!(df.column("str_col")?.dtype(), &DataType::Utf8);
        Ok(())
    }

    #[test]
    fn test_array_conversions() -> Result<()> {
        // Create a Polars array
        let series = Series::new("test", &[1i32, 2, 3]);
        let polars_array = series.to_arrow(POLARS_ARROW_FLAVOR);

        // Convert Polars array to Arrow-rs array
        let arrow_array = convert_polars_arrow_array_to_arrow_rs_array(
            polars_array,
            ArrowDataType::Int32,
        )?;

        // Convert back to Polars array
        let polars_array_back = convert_arrow_rs_array_to_polars_arrow_array(
            &arrow_array,
            polars::datatypes::ArrowDataType::Int32,
        )?;

        // Verify the conversion preserved the data
        let series_back = Series::from_arrow("test", polars_array_back)?;
        assert_eq!(series, series_back);
        Ok(())
    }

    #[test]
    fn test_field_conversions() -> Result<()> {
        // Create a Polars field
        let polars_field = polars_arrow::datatypes::Field::new(
            "test",
            polars::datatypes::ArrowDataType::Int32,
            true,
        );

        // Convert Polars field to Arrow-rs field
        let arrow_field = convert_polars_arrow_field_to_arrow_rs_field(polars_field.clone())?;

        // Convert back to Polars field
        let polars_field_back = convert_arrow_rs_field_to_polars_arrow_field(&arrow_field)?;

        // Verify the conversion preserved the field properties
        assert_eq!(polars_field.name, polars_field_back.name);
        assert_eq!(polars_field.data_type(), polars_field_back.data_type());
        assert_eq!(polars_field.is_nullable(), polars_field_back.is_nullable());
        Ok(())
    }
}
