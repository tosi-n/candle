# OpenAI privacy-filter / OpenMed privacy-filter-nemotron

PII detection via the `openai_privacy_filter` token classifier:

- 1.4B parameter sparse-MoE (8 layers × 128 experts × top-4)
- 50M active parameters per token
- 221 BIOES labels over 55 PII categories
- Constrained Viterbi decoding by default

## Run

```bash
# Default model: OpenMed/privacy-filter-nemotron (Apache 2.0)
cargo run --example openai-privacy-filter --release -- \
    --text "Patient Sarah Johnson, MRN 4872910, phone 415-555-0123."

# Use a local checkout of the weights
cargo run --example openai-privacy-filter --release -- \
    --model-path ~/hybrie-mounts/OpenMed/privacy-filter-nemotron \
    --text "Card 4111-1111-1111-1111 and IBAN GB82WEST12345698765432"

# Plain argmax instead of constrained Viterbi
cargo run --example openai-privacy-filter --release -- --argmax \
    --text "..."
```

## Output format

Each detected span is printed as:

```
  [start..end] class=<idx> text="<original substring>"
```

`class` is the 0-indexed PII category. The 55 class names (in order) are:

```
account_number, age, api_key, bank_routing_number, biometric_identifier,
blood_type, certificate_license_number, city, company_name, coordinate,
country, county, credit_debit_card, customer_id, cvv, date,
date_of_birth, date_time, device_identifier, education_level, email,
employee_id, employment_status, fax_number, first_name, gender,
health_plan_beneficiary_number, http_cookie, ipv4, ipv6, language,
last_name, license_plate, mac_address, medical_record_number,
national_id, occupation, password, phone_number, pin, political_view,
postcode, race_ethnicity, religious_belief, sexuality, ssn, state,
street_address, swift_bic, tax_id, time, unique_id, url, user_name,
vehicle_identifier
```
