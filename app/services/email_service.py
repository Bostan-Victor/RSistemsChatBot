from __future__ import annotations

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_demo_confirmation(
    *,
    to_email: str,
    name: str,
    business_type: str,
    reserved_datetime: datetime,
) -> None:
    """Send a demo reservation confirmation email via Gmail SMTP.

    Requires GMAIL_SMTP_USER and GMAIL_SMTP_APP_PASSWORD env vars.
    Raises ValueError if credentials are missing.
    Raises smtplib.SMTPException on send failure.
    """
    smtp_user = os.getenv("GMAIL_SMTP_USER", "").strip()
    smtp_password = os.getenv("GMAIL_SMTP_APP_PASSWORD", "").strip()

    if not smtp_user or not smtp_password:
        raise ValueError(
            "GMAIL_SMTP_USER and GMAIL_SMTP_APP_PASSWORD must be set to send emails."
        )

    formatted_dt = reserved_datetime.strftime("%d %B %Y, %H:%M")

    subject = "Confirmare rezervare demo — RSistems"
    body_html = f"""
    <html><body style="font-family: Arial, sans-serif; color: #222; max-width: 560px;">
      <h2 style="color: #1f6feb;">Rezervarea dvs. a fost confirmată!</h2>
      <p>Bună, <strong>{name}</strong>,</p>
      <p>Demo-ul dvs. în showroom-ul RSistems a fost rezervat cu succes.</p>
      <table style="border-collapse:collapse; margin: 16px 0;">
        <tr><td style="padding: 6px 16px 6px 0; color: #555;">Data și ora:</td>
            <td style="padding: 6px 0;"><strong>{formatted_dt}</strong></td></tr>
        <tr><td style="padding: 6px 16px 6px 0; color: #555;">Tip afacere:</td>
            <td style="padding: 6px 0;">{business_type}</td></tr>
      </table>
      <p>Un consultant RSistems vă va întâmpina la sediul nostru la ora programată.</p>
      <p style="margin-top: 24px; color: #888; font-size: 13px;">
        Dacă doriți să modificați sau să anulați rezervarea, ne puteți contacta direct.
      </p>
      <hr style="border:none; border-top: 1px solid #eee; margin: 24px 0;">
      <p style="font-size: 12px; color: #aaa;">RSistems — Software pentru HoReCa</p>
    </body></html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg.attach(MIMEText(body_html, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, to_email, msg.as_string())
