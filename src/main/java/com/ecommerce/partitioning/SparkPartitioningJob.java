package com.ecommerce.partitioning;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SparkPartitioningJob {
	private static final Logger logger = LoggerFactory.getLogger(SparkPartitioningJob.class);
	private static final String EXTRACTED_DIR = "data/extracted";
	private static final String ZIP_DIR = "data/partitioned-zip";

	// Table classifications
	private static final Set<String> FACT_TABLES = Set.of("orders", "order_items", "order_payments", "order_reviews");

	private static final Set<String> DIMENSION_TABLES = Set.of("customers", "products", "sellers", "geolocation",
			"product_category_name_translation");

	// All tables (for ZIP completeness)
	private static final Set<String> ALL_TABLES = Set.of("orders", "order_items", "order_payments", "order_reviews",
			"customers", "products", "sellers", "geolocation", "product_category_name_translation");

	// Raw file names
	private static final Map<String, String> TABLE_TO_FILE = Map.ofEntries(
			Map.entry("orders", "olist_orders_dataset.csv"), Map.entry("order_items", "olist_order_items_dataset.csv"),
			Map.entry("order_payments", "olist_order_payments_dataset.csv"),
			Map.entry("order_reviews", "olist_order_reviews_dataset.csv"),
			Map.entry("customers", "olist_customers_dataset.csv"), Map.entry("products", "olist_products_dataset.csv"),
			Map.entry("sellers", "olist_sellers_dataset.csv"),
			Map.entry("geolocation", "olist_geolocation_dataset.csv"),
			Map.entry("product_category_name_translation", "product_category_name_translation.csv"));

	// Table headers (critical for empty files)
	private static final Map<String, String> TABLE_HEADERS = Map.ofEntries(Map.entry("orders",
			"order_id,customer_id,order_status,order_purchase_timestamp,order_approved_at,order_delivered_carrier_date,order_delivered_customer_date,order_estimated_delivery_date\n"),
			Map.entry("order_items",
					"order_id,order_item_id,product_id,seller_id,shipping_limit_date,price,freight_value\n"),
			Map.entry("order_payments",
					"order_id,payment_sequential,payment_type,payment_installments,payment_value\n"),
			Map.entry("order_reviews",
					"review_id,order_id,review_score,review_comment_title,review_comment_message,review_creation_date,review_answer_timestamp\n"),
			Map.entry("customers",
					"customer_id,customer_unique_id,customer_zip_code_prefix,customer_city,customer_state\n"),
			Map.entry("products",
					"product_id,product_category_name,product_name_lenght,product_description_lenght,product_photos_qty,product_weight_g,product_length_cm,product_height_cm,product_width_cm\n"),
			Map.entry("sellers", "seller_id,seller_zip_code_prefix,seller_city,seller_state\n"),
			Map.entry("geolocation",
					"geolocation_zip_code_prefix,geolocation_lat,geolocation_lng,geolocation_city,geolocation_state\n"),
			Map.entry("product_category_name_translation", "product_category_name,product_category_name_english\n"));

	public static void main(String[] args) {
		logger.info("⚡ Starting Spark Partitioning Job");

		// Ensure ZIP directory exists
		try {
			Files.createDirectories(Paths.get(ZIP_DIR));
		} catch (IOException e) {
			logger.error("❌ Failed to create ZIP directory: {}", ZIP_DIR, e);
			return;
		}

		SparkSession spark = SparkSession.builder().appName("DataPartitioningJob").master("local")
				.config("spark.sql.session.timeZone", "UTC")
			    .config("spark.driver.memory", "384m")
			    .config("spark.executor.memory", "384m")
			    .config("spark.driver.maxResultSize", "128m")
			    .config("spark.default.parallelism", "2")
			    .config("spark.sql.adaptive.enabled", "true").config("spark.driver.host", "127.0.0.1")
				.config("spark.driver.bindAddress", "127.0.0.1").getOrCreate();

		try {

			// Get all unique order dates from orders table
			Dataset<Row> orders = spark.read().option("header", "true").option("inferSchema", "true")
					.csv(EXTRACTED_DIR + "/olist_orders_dataset.csv").withColumn("order_date",
							org.apache.spark.sql.functions.to_date(
									org.apache.spark.sql.functions.col("order_purchase_timestamp"),
									"yyyy-MM-dd HH:mm:ss"));

			List<LocalDate> dates = orders.select("order_date").dropDuplicates().collectAsList().stream()
					.map(row -> row.getDate(0).toLocalDate()).collect(Collectors.toList());

			logger.info("🗓️ Found {} unique order dates", dates.size());

			// Load all dimension tables once (full tables in memory)
			Map<String, Dataset<Row>> dimensionTables = loadDimensionTables(spark);
			createZipForDims(spark, dimensionTables);

			// For each date: build ZIP directly (no intermediate files)
			for (LocalDate date : dates) {
				String dateStr = date.format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
				logger.info("📦 Building ZIP for date: {}", dateStr);
				createZipForDate(spark, date, dateStr);
			}

			logger.info("✅ Partitioning and ZIP creation completed successfully!");

		} finally {
			spark.stop();
		}
	}

	private static void createZipForDims(SparkSession spark, Map<String, Dataset<Row>> dimensionTables) {
		for (String dimTable : DIMENSION_TABLES) {

			Path zipPath = Paths.get(ZIP_DIR, dimTable + ".zip");

			try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(zipPath))) {
				Dataset<Row> dimDf = dimensionTables.get(dimTable);
				writeDataFrameToZip(dimDf, dimTable, zos);

			} catch (IOException e) {
				logger.error("❌ Failed to create ZIP for {}", dimTable, e);
			}
		}
	}

	private static Map<String, Dataset<Row>> loadDimensionTables(SparkSession spark) {
		logger.info("📚 Loading dimension tables (full tables)...");
		Map<String, Dataset<Row>> dimensions = new HashMap<>();

		for (String dimTable : DIMENSION_TABLES) {
			String fileName = TABLE_TO_FILE.get(dimTable);
			if (fileName != null) {
				Dataset<Row> df = spark.read().option("header", "true").option("inferSchema", "true")
						.csv(EXTRACTED_DIR + "/" + fileName);
				dimensions.put(dimTable, df);
				logger.info("✅ Loaded {} records for dimension: {}", df.count(), dimTable);
			}
		}

		return dimensions;
	}

	private static void createZipForDate(SparkSession spark, LocalDate date, String dateStr) {
		Path zipPath = Paths.get(ZIP_DIR, dateStr + ".zip");

		try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(zipPath))) {
			// Write fact tables (filtered by date)
			for (String factTable : FACT_TABLES) {
				Dataset<Row> factDf = getFactTableForDate(spark, date, factTable);
				writeDataFrameToZip(factDf, factTable, zos);
			}

			// Write dimension tables (full tables)
		} catch (IOException e) {
			logger.error("❌ Failed to create ZIP for {}", dateStr, e);
		}
	}

	private static Dataset<Row> getFactTableForDate(SparkSession spark, LocalDate date, String factTable) {
		String dateStr = date.format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
		String fileName = TABLE_TO_FILE.get(factTable);

		if (fileName == null)
			return null;

		Dataset<Row> df = spark.read().option("header", "true").option("inferSchema", "true")
				.csv(EXTRACTED_DIR + "/" + fileName);

		// Filter facts by date
		if ("orders".equals(factTable)) {
			return df.filter(org.apache.spark.sql.functions
					.to_date(org.apache.spark.sql.functions.col("order_purchase_timestamp"), "yyyy-MM-dd HH:mm:ss")
					.equalTo(dateStr));
		} else if (FACT_TABLES.contains(factTable)) {
			// Join with orders to get date
			Dataset<Row> orders = spark.read().option("header", "true").option("inferSchema", "true")
					.csv(EXTRACTED_DIR + "/olist_orders_dataset.csv")
					.withColumn("order_date", org.apache.spark.sql.functions.to_date(
							org.apache.spark.sql.functions.col("order_purchase_timestamp"), "yyyy-MM-dd HH:mm:ss"))
					.select("order_id", "order_date");

			return df.join(orders, "order_id").filter(org.apache.spark.sql.functions.col("order_date").equalTo(dateStr))
					.drop("order_date");
		}
		return df;
	}

	private static void writeDataFrameToZip(Dataset<Row> df, String tableName, ZipOutputStream zos) {
		if (df == null)
			return;

		try {
			ZipEntry entry = new ZipEntry(tableName + ".csv");
			zos.putNextEntry(entry);

			// Convert DataFrame to CSV string
			String csv = convertDataFrameToCsv(df);
			zos.write(csv.getBytes(StandardCharsets.UTF_8));

			zos.closeEntry();
		} catch (Exception e) {
			logger.warn("⚠️ Failed to write table {} to ZIP", tableName, e);
		}
	}

	private static String convertDataFrameToCsv(Dataset<Row> df) {
		StringBuilder csv = new StringBuilder();

		// Add header
		String[] columns = df.columns();
		csv.append(String.join(",", escapeCsvColumns(columns))).append("\n");

		// Add rows
		for (Row row : df.collectAsList()) {
			String[] values = new String[columns.length];
			for (int i = 0; i < columns.length; i++) {
				Object value = row.get(i);
				values[i] = (value == null) ? "" : escapeCsvValue(value.toString());
			}
			csv.append(String.join(",", values)).append("\n");
		}

		return csv.toString();
	}

	private static String[] escapeCsvColumns(String[] columns) {
		return Arrays.stream(columns)
				.map(col -> col.contains(",") || col.contains("\"") ? "\"" + col.replace("\"", "\"\"") + "\"" : col)
				.toArray(String[]::new);
	}

	private static String escapeCsvValue(String value) {
		if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
			return "\"" + value.replace("\"", "\"\"") + "\"";
		}
		return value;
	}
}