import os
import random
import string
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)

WORK_SUBJECTS = [
    "Action Required: {project} Deadline - {day}",
    "Q{q} Review Meeting - {day} at {time}",
    "Update: {project} milestone status",
    "Please review: {project} proposal by EOD",
    "Team sync - {day} agenda attached",
    "Follow-up on {project} deliverables",
    "URGENT: Client presentation moved to {day}",
    "Feedback needed on {project} draft",
    "Weekly standup notes - {date}",
    "Performance review schedule: {date}",
]

WORK_BODIES = [
    (
        "Hi team, I wanted to flag that the {project} deliverables are due on {day}. "
        "Please submit your sections to {name} before 5 PM. If you have any blockers, "
        "escalate them immediately so we can resolve them before the deadline. "
        "The client is expecting a polished presentation and we cannot afford delays. "
        "Please confirm receipt of this email and your status update by {day} morning. "
        "Action items: finalize slides, review data, send approval to {name}."
    ),
    (
        "Following up on our Q{q} planning session. We need to finalize the roadmap by {day}. "
        "{name} will be leading the review. Please make sure all dependencies are documented "
        "and risks are flagged in the shared tracker. The engineering team should complete "
        "their estimates by Wednesday. Reminder: no code freeze exceptions without manager sign-off. "
        "Please complete the task assigned to you and respond to this thread confirming completion."
    ),
    (
        "Team, quarterly business review is scheduled for {day} at {time}. "
        "Please prepare a 5-minute update on your current projects, blockers, and next steps. "
        "Attendance is mandatory. The executive team will be present. "
        "Slides must be submitted to {name} by {day} morning for final review. "
        "Action required: update your section of the shared deck and send your approval."
    ),
]

SPAM_SUBJECTS = [
    "Congratulations! You've been selected for a FREE {prize}",
    "Your account needs immediate verification",
    "You have an unclaimed reward - act now!",
    "LIMITED OFFER: Get {product} for just $1 - today only!",
    "URGENT: Your payment is overdue - verify now",
    "You've won a {prize}! Claim within 24 hours",
    "Hot deal: {discount}% OFF everything - ends tonight!",
    "Someone tried to access your account - verify immediately",
]

SPAM_BODIES = [
    (
        "Dear valued customer, CONGRATULATIONS! Our records show that you have been "
        "randomly selected to receive a free {prize} worth over ${val}. "
        "To claim your reward, simply click the link below and enter your personal details. "
        "This offer expires in 24 hours so do not delay. "
        "Click here: http://totally-legit-prize.net/claim?id=38291 "
        "If you do not respond within 24 hours your prize will be forfeited. "
        "Act now! Limited slots available."
    ),
    (
        "ALERT: We have detected suspicious activity on your account. "
        "Your account will be suspended unless you verify your identity immediately. "
        "Click the secure link below to confirm your information: "
        "http://secure-verification-portal.xyz/login "
        "Failure to verify within 12 hours will result in permanent account suspension. "
        "This is an automated security notice. Do not ignore this message. "
        "Enter your username, password, and SSN to complete verification."
    ),
]

NEWSLETTER_SUBJECTS = [
    "This Week in {topic}: Top 5 stories you need to read",
    "{topic} Weekly Digest - {date}",
    "The {topic} newsletter: {headline}",
    "Your {day} briefing: {topic} trends and insights",
    "Must-read: {count} {topic} articles from this week",
]

NEWSLETTER_BODIES = [
    (
        "Welcome to this week's {topic} digest! Here are the top stories making waves: "
        "1. {headline_a} - researchers at MIT have published a new study showing significant "
        "improvements in {area}. The implications for the industry are profound. "
        "2. {headline_b} - a new framework promises to cut development time by 40%. "
        "3. {topic} funding hits record $2.3B this quarter. "
        "4. Top 10 {area} tools. "
        "5. Opinion: Why {topic} will reshape how we work. "
        "That's all for this week. Forward to a colleague who'd enjoy this!"
    ),
    (
        "Hey there! Hope your week is going well. Here's your curated {topic} roundup: "
        "• {headline_a}: The latest benchmark results are in and the numbers are impressive. "
        "• {headline_b}: A developer's perspective on the new {area} release. "
        "• Deep dive: Understanding the economics of {topic} adoption in enterprise. "
        "• Tutorial spotlight: Build a {area} pipeline in under 30 minutes. "
        "Read the full stories on our website. Unsubscribe anytime below."
    ),
]

PERSONAL_SUBJECTS = [
    "Hey, are we still on for {day}?",
    "Quick catch-up this week?",
    "Re: {topic} - thoughts?",
    "Happy birthday {name}!",
    "Saw this and thought of you",
    "Plans this {day}?",
    "Long time no talk!",
    "Checking in",
]

PERSONAL_BODIES = [
    (
        "Hey! Hope you're doing well. Just wanted to check if we're still on for {day}. "
        "I was thinking we could grab coffee around {time} at that place you mentioned. "
        "Let me know if that still works or if you need to reschedule. "
        "Also, did you get a chance to check out that {topic} documentary? "
        "Would love to hear what you think! No rush, text me whenever."
    ),
    (
        "Hi! It's been a while since we properly caught up. "
        "Life has been super busy lately with {topic} and everything. "
        "How are things on your end? Would love to get together sometime soon. "
        "Maybe next {day}? I'm pretty flexible in the afternoon. "
        "Also wanted to say congrats on the {topic} news - so happy for you! "
        "Let me know what works. Miss you!"
    ),
]

FINANCE_SUBJECTS = [
    "Invoice #{inv} Due - {day}",
    "Your statement is ready: {month} {year}",
    "Payment confirmation: ${amount}",
    "Action Required: Outstanding balance ${amount}",
    "Expense report #{inv} needs your approval",
    "Budget alert: {dept} has exceeded threshold",
    "Tax document ready for {year}",
    "Wire transfer confirmation - Ref #{inv}",
]

FINANCE_BODIES = [
    (
        "Dear {name}, please find attached Invoice #{inv} for services rendered in {month}. "
        "Total amount due: ${amount}. Payment is due by {day}. "
        "Bank transfer details: Account #{acct}, Routing #{routing}. "
        "Please reference Invoice #{inv} in your payment description. "
        "Late payments will incur a {pct}% monthly fee. "
        "If you have any questions regarding this invoice please contact {name} immediately. "
        "Thank you for your prompt attention to this matter."
    ),
    (
        "Hi, this is a reminder that your account balance of ${amount} is due on {day}. "
        "To avoid late payment fees, please ensure payment is made before {day} 11:59 PM. "
        "You can pay via bank transfer, credit card, or cheque. "
        "Please log into your account portal to view the full breakdown of charges. "
        "If you believe this charge is incorrect, contact our billing team within 5 business days. "
        "Reference number: INV-{inv}."
    ),
]

ALERT_SUBJECTS = [
    "CRITICAL: {service} is DOWN - immediate action required",
    "[ALERT] CPU usage at {pct}% on {server}",
    "Build #{build} FAILED - {branch}",
    "Security alert: {count} failed login attempts",
    "[WARNING] Disk usage at {pct}% on {server}",
    "Deployment {env} completed successfully",
    "Database backup failed: {db}",
    "API error rate spike: {pct}% over threshold",
]

ALERT_BODIES = [
    (
        "CRITICAL ALERT - {service} has been unreachable for {mins} minutes. "
        "Error: {error}. Last successful ping: {time}. "
        "Affected region: {region}. Estimated user impact: {users} users. "
        "Runbook: https://wiki.internal/runbooks/{service}-outage "
        "On-call engineer: please acknowledge within 5 minutes. "
        "Auto-scaling has been triggered but the issue persists. "
        "Escalation path: L1 -> L2 -> {name}. "
        "Current status: INVESTIGATING. All hands required immediately."
    ),
    (
        "Build #{build} failed on branch {branch} at {time}. "
        "Stage: {stage}. Exit code: {code}. "
        "Error log snippet: AssertionError at line 247 in test_{module}.py. "
        "3 of 14 tests failed. Coverage dropped to {pct}%. "
        "Triggered by commit: {hash}. Author: {name}. "
        "Action required: investigate failing tests and push a fix. "
        "Pipeline is blocked. No deployments can proceed until tests pass. "
        "View full logs: https://ci.internal/build/{build}"
    ),
]


PRIORITY_MAP = {
    "work":       ["high", "high", "medium", "medium", "low"],
    "spam":       ["low", "low", "low"],
    "newsletter": ["low", "low", "medium"],
    "personal":   ["low", "medium", "medium"],
    "finance":    ["high", "high", "medium"],
    "alerts":     ["high", "high", "high", "medium"],
}

PROJECTS    = ["Phoenix", "Atlas", "Horizon", "Nexus", "Orion", "Vega", "Titan", "Aurora"]
NAMES       = ["Alex", "Jordan", "Taylor", "Morgan", "Riley", "Casey", "Sam", "Dana"]
DAYS        = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIMES       = ["9:00 AM", "10:00 AM", "2:00 PM", "3:30 PM", "4:00 PM"]
TOPICS      = ["AI", "Machine Learning", "Data Science", "Cloud", "DevOps", "Fintech", "Crypto"]
HEADLINES_A = [
    "New LLM outperforms GPT-4 on coding benchmarks",
    "Startup raises $500M Series C",
    "Open-source model achieves SOTA on ImageNet",
    "EU passes landmark regulation",
]
HEADLINES_B = [
    "Developer productivity up 35% with new tooling",
    "Cloud costs slashed by containerisation",
    "Framework hits 1M GitHub stars",
    "Remote teams outperform in-office counterparts",
]
AREAS       = ["NLP", "computer vision", "reinforcement learning", "MLOps", "data pipeline"]
SERVICES    = ["API Gateway", "Auth Service", "Payment Service", "Database", "CDN", "Cache Layer"]
SERVERS     = ["prod-web-01", "prod-db-02", "staging-api-03", "us-east-1"]
ERRORS      = ["Connection timed out", "500 Internal Server Error", "OOM killed", "Disk full"]
REGIONS     = ["us-east-1", "eu-west-1", "ap-south-1"]
PRIZES      = ["iPhone 15 Pro", "MacBook Air", "$500 Amazon gift card", "Tesla Model 3"]
PRODUCTS    = ["VPN subscription", "premium antivirus", "cloud storage"]
DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Finance", "Operations"]
MONTHS      = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]


def _rand(lst):
    return random.choice(lst)


def _fake_email_row(category):
    day   = _rand(DAYS)
    time  = _rand(TIMES)
    name  = _rand(NAMES)
    proj  = _rand(PROJECTS)
    q     = random.randint(1, 4)
    inv   = random.randint(10000, 99999)
    build = random.randint(1000, 9999)
    amount = f"{random.randint(500, 25000):,}"
    pct   = random.randint(75, 99)
    topic = _rand(TOPICS)
    date  = f"{_rand(MONTHS)} {random.randint(1, 28)}, {random.randint(2024, 2025)}"
    year  = random.randint(2023, 2025)
    month = _rand(MONTHS)
    acct  = f"{random.randint(10000000, 99999999)}"
    routing = f"{random.randint(100000000, 999999999)}"
    branch = _rand(["main", "develop", "feature/auth", "release/v2.1", "hotfix/payment"])
    stage  = _rand(["lint", "unit-tests", "integration", "build", "deploy"])
    module = _rand(["auth", "payment", "core", "utils", "api"])
    code   = _rand([1, 2, 127, 137])
    hash_  = hex(random.randint(0xAAAAAAA, 0xFFFFFFF))[2:]
    users  = f"{random.randint(100, 50000):,}"
    mins   = random.randint(2, 45)
    count  = random.randint(3, 200)
    prize  = _rand(PRIZES)
    product = _rand(PRODUCTS)
    discount = random.randint(20, 70)
    val   = random.randint(200, 2000)
    dept  = _rand(DEPARTMENTS)
    headline_a = _rand(HEADLINES_A)
    headline_b = _rand(HEADLINES_B)
    area  = _rand(AREAS)
    service = _rand(SERVICES)
    server  = _rand(SERVERS)
    error   = _rand(ERRORS)
    region  = _rand(REGIONS)

    fmts = dict(
        project=proj, day=day, time=time, name=name, q=q, inv=inv, build=build,
        amount=amount, pct=pct, topic=topic, date=date, year=year, month=month,
        acct=acct, routing=routing, branch=branch, stage=stage, module=module,
        code=code, hash=hash_, users=users, mins=mins, count=count, prize=prize,
        product=product, discount=discount, val=val, dept=dept,
        headline=_rand(HEADLINES_A), headline_a=headline_a, headline_b=headline_b, area=area,
        service=service, server=server, error=error, region=region,
        db=_rand(["postgres-prod", "mongo-analytics", "redis-session"]),
        env=_rand(["production", "staging", "canary"]),
    )

    cat_map = {
        "work":       (WORK_SUBJECTS,       WORK_BODIES),
        "spam":       (SPAM_SUBJECTS,       SPAM_BODIES),
        "newsletter": (NEWSLETTER_SUBJECTS, NEWSLETTER_BODIES),
        "personal":   (PERSONAL_SUBJECTS,   PERSONAL_BODIES),
        "finance":    (FINANCE_SUBJECTS,    FINANCE_BODIES),
        "alerts":     (ALERT_SUBJECTS,      ALERT_BODIES),
    }

    subjects, bodies = cat_map[category]

    def safe_format(tmpl, vals):
        formatter = string.Formatter()
        used_keys = {fname for _, fname, _, _ in formatter.parse(tmpl) if fname}
        return tmpl.format(**{k: v for k, v in vals.items() if k in used_keys})

    subject = safe_format(_rand(subjects), fmts)
    body    = safe_format(_rand(bodies),   fmts)

    priority = random.choice(PRIORITY_MAP[category])

    base_date = datetime(2024, 1, 1)
    received_at = base_date + timedelta(days=random.randint(0, 450),
                                        hours=random.randint(6, 22),
                                        minutes=random.randint(0, 59))

    return {
        "email_id":    f"EML-{random.randint(100000, 999999)}",
        "subject":     subject,
        "body":        body,
        "category":    category,
        "priority":    priority,
        "received_at": received_at.isoformat(),
        "word_count":  len(body.split()),
    }


def generate_dataset(n_samples=2000, save_path="data/emails.csv"):
    categories = ["work", "spam", "newsletter", "personal", "finance", "alerts"]
    per_cat    = n_samples // len(categories)
    remainder  = n_samples % len(categories)

    rows = []
    for i, cat in enumerate(categories):
        n = per_cat + (1 if i < remainder else 0)
        for _ in range(n):
            rows.append(_fake_email_row(cat))

    random.shuffle(rows)
    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Dataset saved to {save_path} ({len(df)} rows)")
    print(df["category"].value_counts().to_string())
    return df


if __name__ == "__main__":
    generate_dataset()
