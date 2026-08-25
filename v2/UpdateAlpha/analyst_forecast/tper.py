class TPERFactor(_ConsensusFactor):
    meta = AlphaMeta("tper", "consensus target price / current price - 1")
    dependencies = (
        "rpt_forecast_stk", "rpt_report_author", "d_essentials/close",
    )
    column = "tper"

    def cross_section(self, snapshot):
        reports = snapshot.reports.with_columns(
            pl.mean_horizontal(
                "target_price_ceiling", "target_price_floor"
            ).alias("target_price")
        ).filter(pl.col("target_price") > 0)
        frame = self.aggregate(reports, "target_price")
        ticks = frame["tick"].to_list()
        close = self.context.local_values(
            self.context.config.close_field, snapshot.asof, ticks
        )
        return frame.with_columns(
            pl.Series("close", close)
        ).with_columns(
            pl.when(
                pl.col("close").is_finite() & (pl.col("close") > 0)
            ).then(
                pl.col("target_price") / pl.col("close") - 1
            ).otherwise(None).alias(self.column)
        ).select("tick", self.column)