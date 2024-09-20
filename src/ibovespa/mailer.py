### gmail mailer function ###

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(period):
    msg = MIMEMultipart()
    msg['From'] = "mpps@cin.ufpe.br"
    msg['To'] = "mpps@cin.ufpe.br"
    msg['Subject'] = "Pibic Code"

    message = f"your code ibovespa {period} finished running"
    email = "mpps@cin.ufpe.br"
    password = "rjhq canr ujew tpms"

    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email, password)
        text = msg.as_string()
        server.sendmail(email, email, text)
        server.quit()
        print('Email sent successfully')
    
    except Exception as e:
        print('Something went wrong...', e)
