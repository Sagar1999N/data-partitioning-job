# Data Partitioning Job

A Spark batch job that partitions the static Olist Brazilian E-commerce dataset by order purchase date, producing a set of ZIP files that simulate a vendor delivering daily data drops. These ZIPs are then served by the `data-provider-api`.

---

## Overview

The job reads the full set of Olist CSV files, extracts every unique order date, and for each date produces a ZIP containing:
- **Fact table slices** — orders, order items, order payments, and order reviews filtered to that date
- **Full dimension tables** — customers, products, sellers, geolocation, and product category translation (included once per ZIP so each package is self-contained)

The output also includes standalone dimension ZIPs (`customers.zip`, `products.zip`, etc.) for reference data endpoints.

---

## System Context

```
data-ingestion-orchestrator
        │  (invokes via spark-submit)
        ▼
data-partitioning-job
        │
        ▼ data/partitioned-zip/
  ┌─────────────────────┐
  │  2017-10-02.zip     │  ← daily fact slices + all dimensions
  │  2017-10-03.zip     │
  │  ...                │
  │  customers.zip      │  ← standalone dimension ZIPs
  │  products.zip       │
  └─────────────────────┘
        │
        ▼
  data-provider-api (serves ZIPs over HTTP)
```

---

## Tech Stack

| Component       | Technology       | Version |
|-----------------|------------------|---------|
| Language        | Java             | 17      |
| Build Tool      | Maven            | 3.x     |
| Processing      | Apache Spark SQL | 3.4.0   |
| Scala compat    | 2.12             | —       |
| Logging         | SLF4J + Log4j2   | 2.0.7 / 2.20.0 |

---

## Project Structure

```
data-partitioning-job/
├── src/main/java/com/ecommerce/partitioning/
│   └── SparkPartitioningJob.java   # Single-class Spark job
├── pom.xml                         # Maven build config
└── target/
    └── data-partitioning-job-1.0.0.jar   # Shaded fat JAR
```

---

## Prerequisites

- Java 17+
- Maven 3.x
- Apache Spark 3.4.0+ installed (with `spark-submit` on PATH)
- Extracted CSV files present in `data/extracted/`

### Required Input CSVs

All files must exist under `data/extracted/`:

```
olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
olist_customers_dataset.csv
olist_products_dataset.csv
olist_sellers_dataset.csv
olist_geolocation_dataset.csv
product_category_name_translation.csv
```

---

## Build

```bash
mvn clean package
```

Produces: `target/data-partitioning-job-1.0.0.jar`

---

## Run

### Via spark-submit (recommended)

```bash
spark-submit \
  --class com.ecommerce.partitioning.SparkPartitioningJob \
  --master local[*] \
  target/data-partitioning-job-1.0.0.jar
```

### Direct Java execution

```bash
java -cp target/data-partitioning-job-1.0.0.jar \
  com.ecommerce.partitioning.SparkPartitioningJob
```

### On a Spark cluster

```bash
spark-submit \
  --class com.ecommerce.partitioning.SparkPartitioningJob \
  --master spark://cluster-master:7077 \
  --deploy-mode cluster \
  target/data-partitioning-job-1.0.0.jar
```

---

## Output Structure

```
data/partitioned-zip/
├── 2016-09-04.zip        # Fact rows for that date + all dimension CSVs
├── 2016-09-05.zip
├── ...
├── 2018-08-29.zip
├── customers.zip         # Full customers dimension
├── products.zip
├── sellers.zip
├── geolocation.zip
└── product_category_name_translation.zip
```

Each date ZIP contains:
- `orders_{date}.csv`
- `order_items_{date}.csv`
- `order_payments_{date}.csv`
- `order_reviews_{date}.csv`
- `customers.csv`
- `products.csv`
- `sellers.csv`
- `geolocation.csv`
- `product_category_name_translation.csv`

---

## Spark Configuration

The job applies these settings at runtime:

| Config                            | Value       |
|-----------------------------------|-------------|
| `spark.sql.session.timeZone`      | UTC         |
| `spark.sql.adaptive.enabled`      | true        |
| `spark.driver.host`               | 127.0.0.1   |
| `spark.driver.bindAddress`        | 127.0.0.1   |
| Master                            | `local[*]`  |

---

## License

Apache 2.0