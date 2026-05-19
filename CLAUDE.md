# Dataset: Bitext Customer Service Tagged Training Dataset

## Purpose
Hybrid synthetic dataset built for fine-tuning LLMs (GPT, Mistral, OpenELM) on the Customer Service domain. Generated via Bitext's NLP/NLG pipeline plus automated Data Labeling, then curated by computational linguists. Intended as a base layer for verticalized assistants — fine-tune on this first, then on a small amount of company-specific data.

## Specs
- **Use case:** Intent Detection
- **Vertical:** Customer Service (intents common across 20 verticals)
- **Size:** 26,872 question/answer pairs (~1,000 per intent), ~3.57M tokens
- **Coverage:** 27 intents across 10 categories, 30 entity/slot types, 12 language-generation tags

## Schema (per row)
- `flags` — language-generation tags (see below)
- `instruction` — user request
- `category` — high-level semantic category
- `intent` — specific intent label
- `response` — example virtual-assistant reply

## Categories → Intents
- **ACCOUNT:** create_account, delete_account, edit_account, switch_account
- **CANCELLATION_FEE:** check_cancellation_fee
- **DELIVERY:** delivery_options
- **FEEDBACK:** complaint, review
- **INVOICE:** check_invoice, get_invoice
- **NEWSLETTER:** newsletter_subscription
- **ORDER:** cancel_order, change_order, place_order
- **PAYMENT:** check_payment_methods, payment_issue
- **REFUND:** check_refund_policy, track_refund
- **SHIPPING_ADDRESS:** change_shipping_address, set_up_shipping_address

## Entities (30 slot types)
Placeholders like `{{Order Number}}`, `{{Invoice Number}}`, `{{Customer Support Email}}`, `{{Customer Support Phone Number}}`, `{{Website URL}}`, `{{Date}}`, `{{Date Range}}`, `{{Delivery City}}`, `{{Delivery Country}}`, `{{Money Amount}}`, `{{Refund Amount}}`, `{{Salutation}}`, `{{Client First/Last Name}}`, `{{Account Type/Category/Change}}`, `{{Profile}} / {{Profile Type}}`, `{{Settings}}`, `{{Online Order/Payment/Navigation Interaction}}`, `{{Online Customer Support Channel}}`, `{{Online Company Portal Info}}`, `{{Live Chat Support}}`, `{{Shipping Cut-off Time}}`, `{{Store Location}}`, `{{Program}}`, `{{Upgrade Account}}`.

## Language Generation Tags (used in `flags`)
**Lexical:** `M` morphological, `L` semantic/synonyms
**Syntactic:** `B` basic, `I` interrogative, `C` coordinated, `N` negation
**Register:** `P` polite, `Q` colloquial, `W` offensive
**Style:** `K` keyword-only, `E` abbreviations, `Z` typos/errors
*(Not used in this dataset: `D` indirect speech, `G` regional, `R` respect structures, `Y` code switching.)*

These tags let you slice the data by linguistic style to target different user profiles (e.g., colloquial-heavy for a sneaker bot, formal/polite for retail banking).
