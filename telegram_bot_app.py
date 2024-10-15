import logging
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
from src.pipelines.Waterlevel_pipeline import WaterLevelModel
from datetime import datetime, timedelta

pipeline = WaterLevelModel(station_code="PPB")
waterlevel_prediction = pipeline.run_prediction_pipeline()
waterlevel_data = pipeline.get_waterlevel()

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Define command handler for /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("Check Flash Flood Level", callback_data='alert')],
        [InlineKeyboardButton("Check Water Level", callback_data='check_water_level')],
        [InlineKeyboardButton("Check Water Flow", callback_data='check_water_flow')],
        [InlineKeyboardButton("Check Rainfall", callback_data='check_rainfall')],
        [InlineKeyboardButton("Get Location Image", callback_data='get_location_image')],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "👋 *Welcome to the Flash Flood Alert Bot!* 🚨\n\n"
        "Stay informed and stay safe! You will receive alerts about flash floods in your area.\n\n"
        "Please choose an option below:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

# Define command handler for /alert
async def alert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if waterlevel_prediction[0] > 8:
        risk_level = 'High Risk'
        alert_color = '🚨'  # Red alert for high risk
    elif 5 < waterlevel_prediction[0] <= 8:
        risk_level = 'Medium Risk'
        alert_color = '⚠️'  # Yellow alert for medium risk
    else:
        risk_level = 'Low Risk'
        alert_color = 'ℹ️'  # Info icon for low risk

    alert_message = (
        f"{alert_color} *Flash Flood Alert!* {alert_color}\n\n"
        f"Risk Level: {risk_level}\n"
        f"Water Level Prediction for {datetime.now() + timedelta(days=1):%Y-%m-%d}: {waterlevel_prediction[0]} m\n\n"
        "Please take precautions:\n"
        "• Avoid low-lying areas\n"
        "• Stay indoors if possible\n"
        "• Follow local safety instructions\n\n"
        "Stay safe!"
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=alert_message, parse_mode='Markdown')

# Placeholder for /check_water_level command
async def check_water_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    water_level = f"🌊 *Water Level For {datetime.now() + timedelta(days=1):%Y-%m-%d}*: {waterlevel_prediction[0]}"
    await context.bot.send_message(chat_id=update.effective_chat.id, text=water_level, parse_mode='Markdown')

# Placeholder for /check_water_flow command
async def check_water_flow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    water_flow = "🚰 *Current Water Flow*: 500 cubic meters per second"  # Example data
    await context.bot.send_message(chat_id=update.effective_chat.id, text=water_flow, parse_mode='Markdown')

# Placeholder for /check_rainfall command
async def check_rainfall(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rainfall = "🌧 *Current Rainfall*: 12 mm in the past hour"  # Example data
    await context.bot.send_message(chat_id=update.effective_chat.id, text=rainfall, parse_mode='Markdown')

# Placeholder for /get_location_image command
async def get_location_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    image_url = "https://live.staticflickr.com/690/22061346823_b315e59592_b.jpg"  # Example image URL
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"🖼 *Location Image*:\n[View Image]({image_url})", parse_mode='Markdown')

# Handle button clicks
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()  # Acknowledge the callback

    # Call the appropriate function based on the button clicked
    if query.data == 'alert':
        await alert(update, context)
    elif query.data == 'check_water_level':
        await check_water_level(update, context)
    elif query.data == 'check_water_flow':
        await check_water_flow(update, context)
    elif query.data == 'check_rainfall':
        await check_rainfall(update, context)
    elif query.data == 'get_location_image':
        await get_location_image(update, context)

def main() -> None:
    # Use ApplicationBuilder instead of Updater
    application = ApplicationBuilder().token("7967245504:AAGahcxA2XOZHVMyP-IiLM6RP5bqGvzSYkY").build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))  # Add this line
    application.add_handler(CommandHandler("alert", alert))
    application.add_handler(CommandHandler("check_water_level", check_water_level))
    application.add_handler(CommandHandler("check_water_flow", check_water_flow))
    application.add_handler(CommandHandler("check_rainfall", check_rainfall))
    application.add_handler(CommandHandler("get_location_image", get_location_image))

    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    main()
