import pandera.pandas as pa
from pandera import Check, Column, DataFrameSchema

from src.utils.logger import ProjectLogger

logger = ProjectLogger(__name__).get_logger()

class FraudSchemaValidator:
    """Validate the Gold fraud dataset with Pandera."""

    def __init__(self) -> None:
        self.schema = self._build_schema()

    def _build_schema(self) -> DataFrameSchema:
        return DataFrameSchema(
            columns={
                # ─── Critical columns ───
                "transaction_id": Column(int, nullable=False, unique=True), 
                "transaction_dt": Column(int, nullable=False, checks=Check.ge(0)),
                "amount": Column(float, nullable=False, checks=Check.ge(0)),
                "is_fraud": Column("int32",  nullable=False, checks=Check.isin([0, 1])),
                "uid": Column(str, nullable=False),
                "uid_is_known": Column("int32",  nullable=False, checks=Check.isin([0, 1])),

                # ─── Base features ───
                "product_type": Column(str, nullable=True),
                "card_id": Column("int32", nullable=True), 
                "card_bank_1": Column(float, nullable=True),
                "card_bank_2": Column(float, nullable=True),
                "card_network": Column(str, nullable=True),
                "card_phone_country": Column(float, nullable=True),
                "card_category": Column(str, nullable=True),
                "buyer_region": Column(str, nullable=True),
                "buyer_country_code": Column(str, nullable=True),
                "distance_buyer_merchant": Column(float,    nullable=True),
                "distance_secondary": Column(float, nullable=True),
                "buyer_email_domain": Column(str, nullable=True),
                "recipient_email_domain": Column(str, nullable=True),
                "device_os": Column(str, nullable=True),
                "browser": Column(str, nullable=True),
                "screen_resolution": Column(str, nullable=True),

                # ─── Engineered features ───
                "uid_transaction_count":Column(int, nullable=False),
                "uid_amount_mean": Column(float, nullable=True),
                "uid_amount_std": Column(float,nullable=True),
                "uid_amount_max": Column(float, nullable=True),
                "uid_amount_sum":  Column(float, nullable=True),
                "uid_d10_mean": Column(float, nullable=True),
                "uid_d15_mean": Column(float, nullable=True),
                "velocity_count_1h": Column(int, nullable=False),
                "velocity_count_24h": Column(int, nullable=False),
                "velocity_count_7d": Column(int, nullable=False),
                "velocity_amount_sum_24h": Column(float, nullable=True),
                "velocity_amount_sum_7d":  Column(float, nullable=True),
                "hour_of_day": Column(int, nullable=False),
                "day_of_week": Column(int, nullable=False),
                "is_weekend":  Column("int32",  nullable=False, checks=Check.isin([0, 1])),

                # ─── Device + null flags ───
                "device_family": Column(str, nullable=True),
                "is_null_dist1":  Column("int32",  nullable=True, checks=Check.isin([0, 1])),
                "is_null_dist2":  Column("int32",  nullable=True, checks=Check.isin([0, 1])),
                "is_null_addr1":  Column("int32",  nullable=True, checks=Check.isin([0, 1])),
                "is_null_device_type": Column("int32",  nullable=True, checks=Check.isin([0, 1])),

                # ─── Regex patterns (Vesta columns) ───
                "^v[0-9]+$": Column(float, nullable=True, regex=True),
                "^c[0-9]+$": Column(float, nullable=True, regex=True),
                "^d[0-9]+$": Column(float, nullable=True, regex=True),
                "^m[0-9]+$": Column(str,   nullable=True, regex=True),
                "^id_(0[1-9]|10|11|13|14|1[7-9]|2[0-2]|2[4-6]|32)$": Column(
                    float, nullable=True, regex=True
                ),
                "^id_(12|15|16|23|27|28|29|30|31|33|34|35|36|37|38)$": Column(
                    str, nullable=True, regex=True
                ),
            },
            strict=False,
            coerce=False,
        )

    def validate(self, df) -> None:
        try:
            self.schema.validate(df, lazy=True)
            logger.info(
                "Pandera validation passed: %d rows, %d columns",
                len(df), len(df.columns),
            )
        except pa.errors.SchemaErrors as err:
            logger.error(
                "Pandera validation failed: %d error(s)",
                len(err.failure_cases),
            )
            logger.error("\n%s", err.failure_cases.head(20).to_string())
            raise