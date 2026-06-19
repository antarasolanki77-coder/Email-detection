"""
Dataset Generator for Email Spam Detection
Generates a diverse CSV dataset with 1000+ labeled email samples.
"""

import csv
import random
import os

# --- SPAM TEMPLATES ---
spam_templates = [
    # Lottery / Prize scams
    "Congratulations! You have won a ${amount} prize! Click here to claim now.",
    "You are the lucky winner of our ${amount} grand prize! Act fast!",
    "WINNER! You've been selected to receive a ${amount} gift card. Claim immediately!",
    "Your email has won ${amount} in our international lottery. Reply to collect.",
    "URGENT: You have won a brand new {item}! Click the link to claim your prize.",
    "Final notice: Claim your ${amount} prize before it expires today!",
    "You're our {ordinal} lucky winner this month! Win ${amount} now.",
    "Amazing news! You qualify for a free {item}. Click to get yours today!",
    # Financial / Money scams
    "Make ${amount} per week working from home! No experience needed. Sign up now.",
    "Earn ${amount} daily with this simple trick. Guaranteed income!",
    "Get rich quick! Invest ${amount} today and earn ${amount2} tomorrow!",
    "URGENT: Your bank account needs verification. Click here immediately.",
    "Your account has been compromised! Verify your identity at {url}.",
    "Free money! Get ${amount} deposited into your account today.",
    "Double your income in 30 days! Secret method revealed. Click now.",
    "Congratulations! You qualify for a ${amount} loan at 0% interest!",
    "Your credit score is excellent! Get a pre-approved ${amount} loan now.",
    # Pharmaceutical spam
    "Buy {drug} online at 80% discount! No prescription needed.",
    "Get {drug} delivered to your door. Lowest prices guaranteed!",
    "Special offer: {drug} for only ${small_amount}! Limited time deal.",
    "Lose {weight}lbs in {days} days! Revolutionary weight loss pill.",
    "Doctor recommended {drug} now available without prescription!",
    # Phishing
    "Your {service} account will be suspended. Verify at {url} immediately.",
    "Important: Update your {service} password now to avoid account closure.",
    "Security alert: Unusual login detected on your {service} account. Verify now.",
    "Your {service} payment failed. Update billing info at {url}.",
    "Action required: Confirm your {service} account details or lose access.",
    # Generic spam
    "Buy now and save up to 90%! Unbelievable deals at {url}.",
    "Hot singles in your area want to meet you! Click here.",
    "FREE {item} for the first 100 customers! Hurry, limited stock!",
    "Exclusive deal: Get {item} for FREE! Just pay shipping.",
    "You've been specially selected for this exclusive offer. Don't miss out!",
    "SALE SALE SALE! Everything must go! Up to 95% off!",
    "Cheap {item} available now. Order today for express delivery!",
    "Click here for a surprise gift! You won't believe what's inside!",
    "This is not a joke! You really won! Claim at {url}.",
    "Dear user, your {item} order is ready. Confirm at {url}.",
    "Act now! This offer expires in {hours} hours!",
    "Unsubscribe? You'll miss out on ${amount} worth of free gifts!",
    "WARNING: Your computer has {count} viruses! Download antivirus now!",
    "Free trial of {service}! No credit card required. Sign up at {url}.",
    "Meet beautiful {gender} in your city tonight! Join free at {url}.",
    "Last chance! Buy one get one free on all {item}!",
    "Reply WIN to this message and get a free {item} instantly!",
    "Your package is waiting! Track it at {url}. Delivery fee: ${small_amount}.",
    "Congratulations {name}! You've earned a free vacation to {destination}!",
    "BREAKING: New government grant of ${amount} available. Apply now!",
    "Work from home and earn ${amount}/hour! No skills needed.",
    "Shocking secret the banks don't want you to know! Click to reveal.",
    "Your {service} subscription gift card code: FREEXXXX. Redeem at {url}.",
    "You have {count} unread messages from singles near you!",
    "Flash sale! {item} at 99% off. Only {count} left in stock!",
]

# --- HAM TEMPLATES ---
ham_templates = [
    # Work emails
    "Hi team, the meeting has been rescheduled to {time} tomorrow. Please update your calendars.",
    "Please find attached the {doc_type} for your review. Let me know if you have any questions.",
    "Can you send me the {doc_type} by end of day? We need it for the client presentation.",
    "The project deadline has been extended to {date}. Please plan accordingly.",
    "I've updated the {doc_type} with the latest changes. Please review and approve.",
    "Let's schedule a call to discuss the {topic} project. Are you free {day}?",
    "Hi, I wanted to follow up on our conversation about the {topic} initiative.",
    "The {event} has been confirmed for {date} at {location}. Please RSVP.",
    "Great work on the {topic} presentation! The client was very impressed.",
    "Could you please review the pull request I submitted for the {topic} module?",
    "Reminder: Team standup at {time}. Please prepare your status updates.",
    "I'll be out of office from {date} to {date2}. Contact {name} for urgent matters.",
    "Thanks for sending the {doc_type}. Everything looks good to me.",
    "The {topic} sprint review is scheduled for {day} at {time}.",
    "Please complete the quarterly performance review by {date}.",
    "HR has updated the leave policy. Check the intranet for details.",
    "The {topic} deployment was successful. No issues reported so far.",
    "Can we reschedule our {time} meeting? I have a conflict.",
    "FYI: The {doc_type} has been approved by management.",
    "Welcome aboard! Your onboarding schedule is attached.",
    # Personal emails
    "Hey! Are you coming to {name}'s birthday party on {day}?",
    "Just checking in. How are you doing? It's been a while since we caught up.",
    "I saw this article about {topic} and thought you might find it interesting.",
    "Thanks for dinner last night! We should do it again sometime.",
    "Can you pick up {item} from the store on your way home?",
    "Happy birthday! Wishing you a wonderful year ahead!",
    "Let me know when you're free this weekend. We could go for a hike.",
    "I'm running a bit late. Will be there in about {minutes} minutes.",
    "Don't forget to call mom today. It's her anniversary.",
    "The kids have a school event on {day}. Can you make it?",
    "Sending you the recipe you asked for. Let me know how it turns out!",
    "Did you watch the {topic} game last night? What a finish!",
    "I found a great {topic} course online. Want to take it together?",
    "Reminder: Our dentist appointment is on {day} at {time}.",
    "We need to renew the car insurance by {date}. Can you handle it?",
    # Academic / School
    "Dear students, the {topic} assignment is due on {date}. No extensions.",
    "The exam schedule has been posted on the portal. Check your dates.",
    "Office hours this week: {day} from {time} to {time2}. Room {room}.",
    "Your grade for the {topic} course has been posted. Check the portal.",
    "The {topic} lecture has been moved to {location} for this week.",
    "Please submit your research proposal by {date}. Format guidelines attached.",
    "The library will be closed on {day} for maintenance.",
    "Registration for next semester opens on {date}. Plan your courses.",
    "The {topic} workshop has {count} seats remaining. Register soon.",
    "Congratulations on completing your {topic} certification!",
    # Notifications (legitimate)
    "Your {service} order #{order_num} has been shipped. Estimated delivery: {date}.",
    "Your monthly {service} statement is ready. Log in to view.",
    "Password changed successfully for your {service} account.",
    "Your {service} subscription will renew on {date}.",
    "Receipt for your {service} purchase of ${small_amount} on {date}.",
    "Your flight {flight} to {destination} departs at {time} on {date}.",
    "Your appointment with Dr. {name} is confirmed for {day} at {time}.",
    "Your {service} account has been successfully created. Welcome!",
    "Reminder: Your {item} warranty expires on {date}.",
    "Your {service} refund of ${small_amount} has been processed.",
    # Newsletters / Updates
    "This week in {topic}: new features, updates, and community highlights.",
    "Our latest blog post on {topic} is now live. Check it out!",
    "Join our upcoming webinar on {topic} on {day} at {time}.",
    "Monthly digest: Top {topic} trends and insights for {month}.",
    "Community update: We've reached {count} members! Thank you for your support.",
]

# --- FILL-IN VALUES ---
amounts = ["500", "1000", "1500", "2000", "5000", "10000", "50000", "100000", "1000000"]
amounts2 = ["5000", "10000", "50000", "100000", "500000"]
small_amounts = ["4.99", "9.99", "14.99", "19.99", "29.99", "49.99"]
items = ["iPhone", "Samsung Galaxy", "iPad", "MacBook", "laptop", "TV", "smartwatch",
         "headphones", "camera", "PS5", "Xbox", "Nintendo Switch", "Rolex watch",
         "designer bag", "diamond ring", "gift card", "voucher"]
drugs = ["Viagra", "Cialis", "Xanax", "Ambien", "weight loss pills", "anti-aging cream",
         "CBD oil", "supplements", "vitamins", "protein powder"]
services = ["PayPal", "Amazon", "Netflix", "Apple", "Google", "Microsoft",
            "Facebook", "Instagram", "Bank of America", "Chase", "Wells Fargo",
            "Spotify", "LinkedIn", "Dropbox", "iCloud"]
urls = ["bit.ly/abc123", "tinyurl.com/xyz", "click-here-now.com/offer",
        "secure-verify.net/login", "prize-claim.org/winner", "www.deals99.com/grab"]
ordinals = ["1st", "2nd", "3rd", "5th", "10th", "100th", "1000th"]
names = ["Sarah", "John", "Mike", "Emily", "David", "Lisa", "Robert", "Jennifer",
         "James", "Jessica", "William", "Ashley", "Daniel", "Amanda"]
destinations = ["Hawaii", "Paris", "Cancun", "Bahamas", "Maldives", "Dubai", "Tokyo",
                "London", "Bali", "Caribbean"]
genders = ["women", "men", "people"]
weights = ["10", "15", "20", "25", "30"]
days_count = ["7", "10", "14", "21", "30"]
hours_list = ["2", "4", "6", "12", "24", "48"]
counts = ["3", "5", "7", "10", "15", "50", "100", "500", "1000"]
topics = ["machine learning", "data science", "project management", "quarterly report",
          "marketing strategy", "budget", "sales", "engineering", "design", "analytics",
          "cloud computing", "security", "Python", "AI", "blockchain", "product launch",
          "customer feedback", "team building", "innovation", "sustainability"]
doc_types = ["quarterly report", "budget proposal", "project plan", "meeting notes",
             "status report", "design document", "requirements document", "invoice",
             "contract", "proposal", "presentation slides", "spreadsheet"]
times = ["9:00 AM", "10:00 AM", "10:30 AM", "11:00 AM", "1:00 PM", "2:00 PM",
         "2:30 PM", "3:00 PM", "3:30 PM", "4:00 PM", "5:00 PM"]
times2 = ["11:00 AM", "12:00 PM", "1:00 PM", "3:00 PM", "4:00 PM", "5:00 PM", "6:00 PM"]
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dates = ["March 15", "April 1", "April 20", "May 5", "May 30", "June 10",
         "June 25", "July 1", "July 15", "August 3", "September 12", "October 20",
         "November 5", "December 1"]
dates2 = ["March 20", "April 8", "May 10", "June 15", "July 5", "August 10"]
locations = ["Conference Room A", "Room 302", "the main hall", "Building B",
             "the auditorium", "Zoom", "Google Meet", "the cafeteria"]
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
order_nums = [str(random.randint(100000, 999999)) for _ in range(20)]
flights = ["AA" + str(random.randint(100, 999)) for _ in range(10)]
rooms = [str(random.randint(100, 500)) for _ in range(10)]
minutes_list = ["5", "10", "15", "20", "30"]
events = ["team lunch", "all-hands meeting", "product demo", "hackathon",
          "training session", "town hall", "offsite", "conference"]

def fill_template(template):
    """Fill in template placeholders with random values."""
    result = template
    replacements = {
        "{amount}": random.choice(amounts),
        "{amount2}": random.choice(amounts2),
        "{small_amount}": random.choice(small_amounts),
        "{item}": random.choice(items),
        "{drug}": random.choice(drugs),
        "{service}": random.choice(services),
        "{url}": random.choice(urls),
        "{ordinal}": random.choice(ordinals),
        "{name}": random.choice(names),
        "{destination}": random.choice(destinations),
        "{gender}": random.choice(genders),
        "{weight}": random.choice(weights),
        "{days}": random.choice(days_count),
        "{hours}": random.choice(hours_list),
        "{count}": random.choice(counts),
        "{topic}": random.choice(topics),
        "{doc_type}": random.choice(doc_types),
        "{time}": random.choice(times),
        "{time2}": random.choice(times2),
        "{day}": random.choice(days),
        "{date}": random.choice(dates),
        "{date2}": random.choice(dates2),
        "{location}": random.choice(locations),
        "{month}": random.choice(months),
        "{order_num}": random.choice(order_nums),
        "{flight}": random.choice(flights),
        "{room}": random.choice(rooms),
        "{minutes}": random.choice(minutes_list),
        "{event}": random.choice(events),
    }
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value, 1)
    # Handle any remaining duplicates of the same placeholder
    for placeholder, value in replacements.items():
        while placeholder in result:
            result = result.replace(placeholder, random.choice(
                amounts if placeholder == "{amount}" else
                dates if placeholder == "{date}" else
                times if placeholder == "{time}" else
                names if placeholder == "{name}" else
                [value]
            ), 1)
    return result


def generate_dataset(output_path, spam_count=550, ham_count=550):
    """Generate the spam/ham dataset CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []

    # Generate spam emails
    for _ in range(spam_count):
        template = random.choice(spam_templates)
        text = fill_template(template)
        rows.append(("spam", text))

    # Generate ham emails
    for _ in range(ham_count):
        template = random.choice(ham_templates)
        text = fill_template(template)
        rows.append(("ham", text))

    # Shuffle
    random.seed(42)
    random.shuffle(rows)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text"])
        writer.writerows(rows)

    print(f"Dataset generated: {len(rows)} emails ({spam_count} spam, {ham_count} ham)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output = os.path.join(project_root, "dataset", "spam.csv")
    generate_dataset(output)
