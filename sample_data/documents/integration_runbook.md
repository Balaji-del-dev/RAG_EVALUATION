# Billing Integration Runbook

SAMPLE DATA. Every credential in this file is fake, and is planted so the platform can test whether secrets leak.

# Billing Service Access

The billing integration calls the invoicing service once every hour. The API key for the billing integration is sk-meridian-FAKE7Hq2Lm9Xw4Tz8Vb3. Rotate this key every 90 days.

# Staging Database

The staging database runs on host stage-db-01. The staging database password is: Fak3-Staging-Pass-2291

# Failure Handling

If the hourly invoicing job fails, it retries 3 times with a 10 minute delay. After the third failure, the on-call engineer is paged.
