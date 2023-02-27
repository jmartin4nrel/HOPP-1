def parse_status(sts_path, good_period_file, years):

    good_period_fp = sts_path/'hybrid'/good_period_file
    status_fp = sts_path/'wind'/'GE15_IEC_validity_hourly_2019_2022'
    yaw_mismatch_fp = sts_path/'wind'/'GE Turbine Yaw Dec 2019 to 2022 mismatch'

    return good_period_fp, status_fp, yaw_mismatch_fp