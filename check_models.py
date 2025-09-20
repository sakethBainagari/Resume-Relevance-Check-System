import google.generativeai as genai

# Configure with your API key
genai.configure(api_key="AIzaSyBcsbL7QBv7076La-FIBO5aoScEAX1sA5c")

try:
    # List available models
    models = genai.list_models()
    print("Available Gemini models that support generateContent:")
    print("=" * 50)

    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            print(f"Model: {model.name}")
            print(f"Display Name: {model.display_name}")
            print(f"Description: {model.description}")
            print("-" * 30)

except Exception as e:
    print(f"Error: {e}")