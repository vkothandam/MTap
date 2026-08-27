import nodemailer from "nodemailer";
import { PHONE_CONFIG } from "../constants.js";

// Sends the OTP entry link to a phone via an email->SMS carrier gateway.
// Ported from MBin scrape/utils/SMSNotifier.js (nodemailer, Gmail transport).

export interface SMSNotifierConfig {
  email: string;
  emailPassword: string;
  phoneNumber?: string;
  carrier?: string;
}

export class SMSNotifier {
  static CARRIERS: Record<string, string> = {
    verizon: "vtext.com",
    tmobile: "tmomail.net",
    att: "txt.att.net",
    sprint: "messaging.sprintpcs.com",
  };

  private config: Required<SMSNotifierConfig>;
  private transporter: nodemailer.Transporter;
  private from: string;

  constructor(config: SMSNotifierConfig) {
    this.config = {
      email: config.email,
      emailPassword: config.emailPassword,
      phoneNumber: config.phoneNumber ?? PHONE_CONFIG.PERSONAL_NUMBER,
      carrier: config.carrier ?? PHONE_CONFIG.CARRIER,
    };
    this.from = config.email;
    this.transporter = nodemailer.createTransport({
      service: "gmail",
      auth: { user: config.email, pass: config.emailPassword },
    });
  }

  async sendNotification(message: string): Promise<boolean> {
    const gateway = SMSNotifier.CARRIERS[this.config.carrier];
    if (!this.config.phoneNumber || !gateway) {
      console.error(
        `[SMSNotifier] Missing phone number or unknown carrier '${this.config.carrier}' — cannot send OTP link`,
      );
      return false;
    }
    const smsEmail = `${this.config.phoneNumber}@${gateway}`;

    try {
      await this.transporter.sendMail({ from: this.from, to: smsEmail, text: message });
      console.log("SMS notification sent successfully");
      return true;
    } catch (error) {
      console.error("Failed to send SMS notification:", error);
      return false;
    }
  }
}
