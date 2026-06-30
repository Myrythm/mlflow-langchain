"""Hand-written evaluation set for the Nimbus support KB.

Each record is shaped for ``mlflow.genai.evaluate``:
  - ``inputs``        -> {"question": <user question>}
  - ``expectations``  -> {"expected_facts": [<key facts a correct answer must contain>],
                          "source_doc": <doc_id the answer is grounded in>}

Facts are grounded in ``docs/nimbus-fact-sheet.md``. Keep this in sync with the articles.
"""

EVAL_DATASET = [
    {
        "inputs": {"question": "How much does the Pro plan cost per member?"},
        "expectations": {
            "expected_facts": [
                "$12 per member/month billed annually",
                "$14 per member/month billed monthly",
            ],
            "source_doc": "pricing-and-billing",
        },
    },
    {
        "inputs": {"question": "How many members can I have on the Free plan?"},
        "expectations": {
            "expected_facts": ["The Free plan supports up to 3 members"],
            "source_doc": "pricing-and-billing",
        },
    },
    {
        "inputs": {"question": "What is the file upload size limit on each plan?"},
        "expectations": {
            "expected_facts": ["250 MB on Pro", "5 GB on Business"],
            "source_doc": "pricing-and-billing",
        },
    },
    {
        "inputs": {"question": "How many automation runs does the Business plan include?"},
        "expectations": {
            "expected_facts": ["10,000 automation runs per month"],
            "source_doc": "pricing-and-billing",
        },
    },
    {
        "inputs": {"question": "Can I get a refund after subscribing?"},
        "expectations": {
            "expected_facts": [
                "14-day money-back guarantee on the first purchase",
                "the refund must be requested within 14 days",
            ],
            "source_doc": "refunds-and-cancellation",
        },
    },
    {
        "inputs": {"question": "What happens to my data when I cancel my subscription?"},
        "expectations": {
            "expected_facts": [
                "cancellation stops auto-renewal",
                "access continues until the end of the current billing period",
            ],
            "source_doc": "refunds-and-cancellation",
        },
    },
    {
        "inputs": {"question": "Does Nimbus support single sign-on?"},
        "expectations": {
            "expected_facts": [
                "Google and Microsoft SSO on the Business plan",
                "SAML SSO on the Enterprise plan",
            ],
            "source_doc": "security-and-sso",
        },
    },
    {
        "inputs": {"question": "How do I turn on two-factor authentication?"},
        "expectations": {
            "expected_facts": [
                "2FA uses a TOTP authenticator app",
                "enable it from account security settings",
            ],
            "source_doc": "security-and-sso",
        },
    },
    {
        "inputs": {"question": "Which apps can I integrate with Nimbus?"},
        "expectations": {
            "expected_facts": ["Slack", "GitHub", "Google Drive"],
            "source_doc": "integrations",
        },
    },
    {
        "inputs": {"question": "How do I connect Slack to Nimbus?"},
        "expectations": {
            "expected_facts": [
                "go to Settings then Integrations",
                "authorize the connection via OAuth",
            ],
            "source_doc": "integrations",
        },
    },
    {
        "inputs": {"question": "How do I invite a teammate to my workspace?"},
        "expectations": {
            "expected_facts": [
                "invite members by email from the workspace members settings",
                "assign them a role",
            ],
            "source_doc": "account-and-teams",
        },
    },
    {
        "inputs": {"question": "What member roles are available?"},
        "expectations": {
            "expected_facts": ["Owner", "Admin", "Member", "Guest"],
            "source_doc": "account-and-teams",
        },
    },
    {
        "inputs": {"question": "How do I create my first board?"},
        "expectations": {
            "expected_facts": [
                "create a workspace first",
                "add a board from the dashboard",
            ],
            "source_doc": "getting-started",
        },
    },
    {
        "inputs": {"question": "My notifications aren't coming through. What should I do?"},
        "expectations": {
            "expected_facts": [
                "check your notification settings",
                "verify email deliverability / check spam",
            ],
            "source_doc": "troubleshooting",
        },
    },
]
