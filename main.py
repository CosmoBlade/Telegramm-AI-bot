from transformers import pipeline
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import logging

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def load_model():
    return pipeline(
        "text-generation",
        model="deepseek-ai/DeepSeek-R1-0528",
        trust_remote_code=True
    )

model_pipe = load_model()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! I am AI bot, based on deep Seek R1, use /ask for your questions"
    
async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.replace('/ask', '').strip()
    
    if not user_input:
        await update.message.reply_text("❌ Write question after /ask!")
        return
    
    try:
        messages = [{"role": "user", "content": user_input}]
        
        response = model_pipe(
            messages,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True
        )
        
        ai_response = response[0]['generated_text']
        await update.message.reply_text(ai_response)
    
    except Exception as e:
        logging.error(f"Error: {e}")
        await update.message.reply_text("⚠️ Error. Try again later")

def main():
    application = Application.builder().token("BOT_TOKEN").build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("ask", ask_command))
    
    application.run_polling()

if __name__ == "__main__":
    main()