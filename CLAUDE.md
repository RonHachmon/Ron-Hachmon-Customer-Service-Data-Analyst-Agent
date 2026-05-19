# Dataset: Bitext Customer Service Tagged Training Dataset

## Purpose
Hybrid synthetic dataset built for fine-tuning LLMs (GPT, Mistral, OpenELM) on the Customer Service domain. Generated via Bitext's NLP/NLG pipeline plus automated Data Labeling, then curated by computational linguists. Intended as a base layer for verticalized assistants — fine-tune on this first, then on a small amount of company-specific data.

## Specs
- **Use case:** Intent Detection
- **Vertical:** Customer Service (intents common across 20 verticals)
- **Size:** 26,872 question/answer pairs, ~3.57M tokens
- **Coverage:** 27 intents across 11 categories, 30 entity/slot types, 12+ language-generation tags


## Schema (per row)
- `flags` — language-generation tags (see below)
- `instruction` — user request
- `category` — high-level semantic category
- `intent` — specific intent label
- `response` — example virtual-assistant reply

## Categories → Intents (with row counts)
- **ACCOUNT** (5,986): create_account, delete_account, edit_account, recover_password, registration_problems, switch_account
- **CANCEL** (950): check_cancellation_fee
- **CONTACT** (1,999): contact_customer_service, contact_human_agent
- **DELIVERY** (1,994): delivery_options, delivery_period
- **FEEDBACK** (1,997): complaint, review
- **INVOICE** (1,999): check_invoice, get_invoice
- **ORDER** (3,988): cancel_order, change_order, place_order, track_order
- **PAYMENT** (1,998): check_payment_methods, payment_issue
- **REFUND** (2,992): check_refund_policy, get_refund, track_refund
- **SHIPPING** (1,970): change_shipping_address, set_up_shipping_address
- **SUBSCRIPTION** (999): newsletter_subscription

Categories are uppercased by `agent/data.py`; intents are kept in their original snake_case form.

## Entities (30 slot types)
Placeholders like `{{Order Number}}`, `{{Invoice Number}}`, `{{Customer Support Email}}`, `{{Customer Support Phone Number}}`, `{{Website URL}}`, `{{Date}}`, `{{Date Range}}`, `{{Delivery City}}`, `{{Delivery Country}}`, `{{Money Amount}}`, `{{Refund Amount}}`, `{{Salutation}}`, `{{Client First/Last Name}}`, `{{Account Type/Category/Change}}`, `{{Profile}} / {{Profile Type}}`, `{{Settings}}`, `{{Online Order/Payment/Navigation Interaction}}`, `{{Online Customer Support Channel}}`, `{{Online Company Portal Info}}`, `{{Live Chat Support}}`, `{{Shipping Cut-off Time}}`, `{{Store Location}}`, `{{Program}}`, `{{Upgrade Account}}`.

## Language Generation Tags (used in `flags`)
**Lexical:** `M` morphological, `L` semantic/synonyms
**Syntactic:** `B` basic, `I` interrogative, `C` coordinated, `N` negation
**Register:** `P` polite, `Q` colloquial, `W` offensive
**Style:** `K` keyword-only, `E` abbreviations, `Z` typos/errors

Tags actually observed in this dataset: `B C E I K L M N P Q S V W Z` — `S` and `V` appear in rows but aren't documented in the original Bitext spec; treat them as data-only signals until clarified.

These tags let you slice the data by linguistic style to target different user profiles (e.g., colloquial-heavy for a sneaker bot, formal/polite for retail banking).
