from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from google import genai
import json
import markdown

# Create Gemini client
client = genai.Client(api_key=settings.GEMINI_API_KEY)

@csrf_exempt   # keep it simple for beginners
def chatbot_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_message = data.get("message", "")

        # Send message to Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_message
        )

        # Get the bot's reply text 
        bot_reply_markdown = response.text 
        
        # Convert the Markdown text to HTML 
        bot_reply_html = markdown.markdown(bot_reply_markdown) 
        return JsonResponse({"reply": bot_reply_html})
    
    # For GET → load the chat page
    return render(request, "chat_app/chat.html")
