from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
messages = ["Win a lottery now",
            "You have won a free meal",
            "I Love Animals",
            "Hey How are you",
            "Here's a free coupon for you",
            "Win a free iPhone now!",
            "Meeting scheduled for tomorrow at 10 AM",
            "Limited-time offer: 70% discount on all items!",
            "Your Amazon order has been shipped",
            "Congratulations! You have been selected for a prize",
            "Weekly project status update",
            "Get rich quick with this secret investment plan",
            "Your OTP for login is 583920",
            "Earn $5000 per week working from home!",
            "Lunch with the client has been postponed",
            "You won’t believe what happens next",
            "Team meeting agenda and notes",
            "Exclusive deal just for you — act fast!",
            "Reminder: doctor appointment tomorrow",
            "Claim your free vacation to Hawaii!",
            "Invoice for your recent purchase",
            "Double your followers overnight!",
            "Minutes of yesterday’s conference call",
            "Urgent notice: update your account details",
            "Happy birthday! Wishing you a wonderful day"]

labeling = [1,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]

vect = CountVectorizer()

x = vect.fit_transform(messages)

x_train, x_test, y_train, y_test = train_test_split(x, labeling, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print("Accuracy",accuracy_score(y_test,y_pred))

test_emails = [
    "Win a brand new iPhone today!",
    "Meeting rescheduled to Monday at 10 AM",
    "Get a free vacation to Hawaii now!",
    "Team lunch invitation for Friday",
    "Claim your exclusive cash prize today!",
    "Project deadline extended to next week",
    "You have won a $500 gift card!",
    "Reminder: Submit your timesheet",
    "Limited-time offer: Buy one get one free",
    "Your invoice for October is attached",
    "Congratulations! You’re our lucky winner!",
    "Weekly status report meeting link",
    "Grab your free coupons before midnight",
    "Dinner plans for tomorrow?",
    "Special discount just for you!",
    "Please review the attached document",
    "Act now to double your income!",
    "Happy birthday! Enjoy your day!",
    "Exclusive offer: Free membership upgrade",
    "Let’s catch up over coffee this weekend",
    "Final notice: Update your account details",
    "Flight itinerary for your upcoming trip",
    "Earn money working from home easily!",
    "Lunch meeting confirmed for Thursday",
    "Win cash rewards instantly by signing up!",
    "Here are your test results",
    "Your parcel is out for delivery",
    "Get rich quick with this simple trick!",
    "Minutes of the last board meeting",
    "Your subscription will expire soon",
    "Congratulations! Free tickets for you!",
    "Reminder: Client meeting at 2 PM",
    "Claim your limited-time reward points!",
    "Your friend sent you a message",
    "You’re pre-approved for a low-interest loan!",
    "Schedule for tomorrow’s training session",
    "Double your followers overnight — click now!",
    "Attached is your travel expense report",
    "Final warning: Account suspended!",
    "Invitation: Company annual day celebration",
    "Free streaming access for one month!",
    "Here’s the summary of today’s discussion",
    "Unlock exclusive deals inside!",
    "Doctor’s appointment confirmation",
    "100% guaranteed weight loss results!",
    "Payment confirmation for your recent purchase",
    "Claim your bonus cash today only!",
    "Join us for a webinar on data science",
    "Limited seats left! Register now!",
    "Weekly newsletter – stay updated",
    "Your bank statement for this month"
]
newValX = vect.transform(test_emails)
predictions = model.predict(newValX)

for msg,pred in zip(test_emails,predictions):
    print(f"'{msg}->","Spam" if pred == 1 else "Ham")

