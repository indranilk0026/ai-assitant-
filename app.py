import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import json
from datetime import datetime
import io
import base64

# Page config
st.set_page_config(
    page_title="AI Wardrobe Assisant",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Fixed
st.markdown("""
    <style>
        .main-header {
            text-align: center;
            color: #667eea;
            font-size: 3em;
            margin-bottom: 10px;
        }
        .sub-header {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .outfit-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            margin: 10px 0;
        }
        .match-score {
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }
        .chat-message {
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .user-message {
            background: #e3f2fd;
            text-align: right;
        }
        .ai-message {
            background: #f3e5f5;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'wardrobe_items' not in st.session_state:
    st.session_state.wardrobe_items = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'upload'
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Fashion trends dictionary for understanding
FASHION_TRENDS = {
    'barbiecore': ['pink', 'feminine', 'playful', 'bright', 'pastel pink', 'hot pink'],
    'cottagecore': ['floral', 'vintage', 'pastoral', 'soft', 'prairie dress', 'natural'],
    'dark academia': ['brown', 'beige', 'vintage', 'scholarly', 'tweed', 'plaid'],
    'streetwear': ['oversized', 'sneakers', 'hoodie', 'urban', 'casual', 'athletic'],
    'minimalist': ['simple', 'clean', 'neutral', 'basic', 'monochrome', 'sleek'],
    'y2k': ['metallic', 'butterfly', 'low-rise', 'colorful', 'nostalgic', '2000s'],
    'coastal grandmother': ['linen', 'relaxed', 'beige', 'coastal', 'effortless', 'light'],
    'clean girl': ['slicked back', 'natural', 'minimalist', 'gold jewelry', 'simple']
}

# Load CLIP model (OpenAI CLIP or fallback to Hugging Face transformers)
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # First try OpenAI's CLIP (clip.load)
    try:
        # If this raises AttributeError (or other), we'll fall back below
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess, device
    except Exception:
        # Try fallback to Hugging Face transformers' CLIP implementation
        try:
            from transformers import CLIPModel, CLIPProcessor

            hf_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            hf_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            hf_model.to(device)
            return hf_model, hf_processor, device
        except Exception as e:
            # Neither OpenAI CLIP nor transformers CLIP are available/working
            # Raise a clear error so the Streamlit UI can show instructions
            raise RuntimeError(
                "Failed to load CLIP. Install the official OpenAI CLIP or Hugging Face transformers. "
                "For the OpenAI CLIP implementation run:\n"
                "  pip uninstall clip -y; pip install git+https://github.com/openai/CLIP.git\n"
                "Or install Hugging Face transformers:\n"
                "  pip install transformers\n"
                f"Underlying error: {e}"
            )

# Extract features from image
def extract_image_features(image, model, preprocess, device):
    """
    Support both OpenAI CLIP (model.encode_image + preprocess as torchvision transform)
    and Hugging Face transformers (CLIPModel + CLIPProcessor).
    """
    # OpenAI CLIP path
    if hasattr(model, "encode_image"):
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    # Hugging Face transformers path
    # preprocess is a CLIPProcessor
    inputs = preprocess(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        # CLIPModel.get_image_features expects pixel_values
        image_features = model.get_image_features({k: v for k, v in inputs.items() if k == 'pixel_values'})
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

# Extract features from text
def extract_text_features(text, model, preprocess, device):
    """
    Support both OpenAI CLIP (model.encode_text + clip.tokenize) and
    Hugging Face transformers (CLIPModel + CLIPProcessor.tokenizer).
    """
    # OpenAI CLIP path
    if hasattr(model, "encode_text"):
        text_input = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

    # Hugging Face transformers path
    # preprocess here is expected to be a CLIPProcessor
    # Use the processor to tokenize
    try:
        # processor = preprocess (CLIPProcessor)
        inputs = preprocess(text=[text], return_tensors="pt", padding=True)
    except Exception:
        # If preprocess isn't a processor, fall back to raising a clear error
        raise RuntimeError("Text preprocessing failed: CLIP processor/tokenizer not available.")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        # CLIPModel.get_text_features expects input_ids and attention_mask
        text_features = model.get_text_features({k: v for k, v in inputs.items() if k in ('input_ids', 'attention_mask')})
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

# Calculate similarity
def calculate_similarity(query_features, item_features):
    return np.dot(query_features, item_features.T)[0][0]

# Enhanced query understanding
def understand_query(query):
    query_lower = query.lower()
    
    # Detect trends
    detected_trends = []
    for trend, keywords in FASHION_TRENDS.items():
        if any(keyword in query_lower for keyword in keywords) or trend in query_lower:
            detected_trends.append(trend)
    
    # Detect occasions
    occasions = {
        'casual': ['casual', 'everyday', 'relaxed', 'comfortable'],
        'formal': ['formal', 'business', 'professional', 'office', 'work'],
        'party': ['party', 'night out', 'club', 'bar', 'evening'],
        'date': ['date', 'romantic', 'dinner'],
        'gym': ['gym', 'workout', 'exercise', 'fitness'],
        'beach': ['beach', 'pool', 'swim', 'vacation']
    }
    
    detected_occasion = 'casual'
    for occasion, keywords in occasions.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_occasion = occasion
            break
    
    # Detect weather preferences
    weather_keywords = {
        'hot': ['hot', 'summer', 'warm', 'sunny'],
        'cold': ['cold', 'winter', 'cool', 'chilly'],
        'rainy': ['rain', 'rainy', 'wet']
    }
    
    detected_weather = None
    for weather, keywords in weather_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_weather = weather
            break
    
    return {
        'trends': detected_trends,
        'occasion': detected_occasion,
        'weather': detected_weather,
        'original_query': query
    }

# Generate outfit recommendations
def generate_outfit_recommendations(query_analysis, wardrobe_items, model, device, preferences):
    if not wardrobe_items:
        return [], []
    
    # Build enhanced query
    query_parts = [query_analysis['original_query']]
    
    if query_analysis['trends']:
        query_parts.extend(query_analysis['trends'])
    
    if preferences['weather']:
        query_parts.append(preferences['weather'])
    
    if preferences['event']:
        query_parts.append(preferences['event'])
    
    enhanced_query = ' '.join(query_parts)
    query_features = extract_text_features(enhanced_query, model, device)
    
    # Score each item
    scored_items = []
    for item in wardrobe_items:
        similarity = calculate_similarity(query_features, item['features'])
        
        # Apply preference bonuses
        bonus = 0
        if preferences['weather'] and preferences['weather'].lower() in item.get('description', '').lower():
            bonus += 0.1
        if preferences['event'] and preferences['event'].lower() in item.get('description', '').lower():
            bonus += 0.1
        
        final_score = min(similarity + bonus, 1.0)
        scored_items.append({
            **item,
            'match_score': final_score
        })
    
    # Sort by score
    scored_items.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Create outfit combinations
    categories = {}
    for item in scored_items:
        category = item.get('category', 'other')
        if category not in categories:
            categories[category] = []
        categories[category].append(item)
    
    # Generate outfit combinations
    outfits = []
    
    # Try to create complete outfits
    if 'top' in categories and 'bottom' in categories:
        for top in categories['top'][:3]:
            for bottom in categories['bottom'][:3]:
                outfit = {
                    'items': [top, bottom],
                    'avg_score': (top['match_score'] + bottom['match_score']) / 2
                }
                
                # Add shoes if available
                if 'shoes' in categories:
                    outfit['items'].append(categories['shoes'][0])
                    outfit['avg_score'] = sum(item['match_score'] for item in outfit['items']) / len(outfit['items'])
                
                # Add accessories if available
                if 'accessories' in categories:
                    outfit['items'].append(categories['accessories'][0])
                    outfit['avg_score'] = sum(item['match_score'] for item in outfit['items']) / len(outfit['items'])
                
                outfits.append(outfit)
    
    # Sort outfits by score
    outfits.sort(key=lambda x: x['avg_score'], reverse=True)
    
    return outfits[:5], scored_items[:10]

# AI Chat Response
def generate_ai_response(query, query_analysis, wardrobe_items):
    trends_text = f" I detected you're interested in {', '.join(query_analysis['trends'])} style!" if query_analysis['trends'] else ""
    occasion_text = f" This is perfect for a {query_analysis['occasion']} occasion."
    
    if not wardrobe_items:
        return f"I'd love to help you with that!{trends_text} However, I don't see any items in your wardrobe yet. Please upload some items first!"
    
    response = f"Great question!{trends_text}{occasion_text} Let me find the perfect outfits from your wardrobe that match your style!"
    
    return response

# Upload page
def upload_page():
    st.markdown("<h1 class='main-header'>AI Wardrobe Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Upload your wardrobe and get personalized outfit recommendations</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Your Wardrobe Items")
        
        uploaded_files = st.file_uploader(
            "Choose images of your clothes",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload clear photos of individual clothing items"
        )
        
        if uploaded_files:
            st.info(f"{len(uploaded_files)} items selected")
            
            # Item details form
            with st.form("item_details"):
                st.write("*Add details for better recommendations:*")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    category = st.selectbox(
                        "Category",
                        ['top', 'bottom', 'dress', 'shoes', 'accessories', 'outerwear', 'other']
                    )
                    
                    color = st.text_input("Primary Color", placeholder="e.g., blue, black, white")
                
                with col_b:
                    season = st.multiselect(
                        "Suitable for",
                        ['spring', 'summer', 'fall', 'winter', 'all-season']
                    )
                    
                    formality = st.select_slider(
                        "Formality Level",
                        options=['casual', 'smart-casual', 'business', 'formal']
                    )
                
                description = st.text_area(
                    "Description (optional)",
                    placeholder="e.g., cotton t-shirt, perfect for summer, comfortable",
                    help="This helps the AI understand your item better"
                )
                
                submitted = st.form_submit_button("Add to Wardrobe", type="primary")
                
                if submitted and uploaded_files:
                    if not st.session_state.model_loaded:
                        with st.spinner("Loading AI model... (first time only)"):
                            model, preprocess, device = load_clip_model()
                            st.session_state.model = model
                            st.session_state.preprocess = preprocess
                            st.session_state.device = device
                            st.session_state.model_loaded = True
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}...")
                        
                        image = Image.open(uploaded_file).convert('RGB')
                        
                        # Extract features
                        features = extract_image_features(
                            image,
                            st.session_state.model,
                            st.session_state.preprocess,
                            st.session_state.device
                        )
                        
                        # Convert image to base64 for storage
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        # Store item
                        item = {
                            'id': len(st.session_state.wardrobe_items),
                            'image': img_str,
                            'category': category,
                            'color': color,
                            'season': season,
                            'formality': formality,
                            'description': description,
                            'features': features,
                            'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M")
                        }
                        
                        st.session_state.wardrobe_items.append(item)
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    status_text.empty()
                    progress_bar.empty()
                    st.success(f"Successfully added {len(uploaded_files)} items to your wardrobe!")
                    st.balloons()
    
    with col2:
        st.subheader("Your Wardrobe Stats")
        
        if st.session_state.wardrobe_items:
            total_items = len(st.session_state.wardrobe_items)
            
            st.metric("Total Items", total_items)
            
            # Category breakdown
            categories = {}
            for item in st.session_state.wardrobe_items:
                cat = item['category']
                categories[cat] = categories.get(cat, 0) + 1
            
            st.write("*By Category:*")
            for cat, count in categories.items():
                st.write(f"• {cat.title()}: {count}")
            
            st.markdown("---")
            
            if st.button("Clear Wardrobe", type="secondary"):
                st.session_state.wardrobe_items = []
                st.rerun()
        else:
            st.info("Your wardrobe is empty. Upload some items to get started!")
    
    # Display uploaded items
    if st.session_state.wardrobe_items:
        st.markdown("---")
        st.subheader("Your Wardrobe")
        
        cols = st.columns(4)
        for idx, item in enumerate(st.session_state.wardrobe_items[-8:]):
            with cols[idx % 4]:
                img_data = base64.b64decode(item['image'])
                img = Image.open(io.BytesIO(img_data))
                st.image(img)
                st.caption(f"{item['category'].title()} • {item['color']}")
    
    # Navigation
    st.markdown("---")
    if st.session_state.wardrobe_items:
        if st.button("Continue to Style Assistant", type="primary", use_container_width=True):
            st.session_state.current_page = 'assistant'
            st.rerun()

# Assistant page
def assistant_page():
    st.markdown("<h1 class='main-header'>Your AI Style Assistant</h1>", unsafe_allow_html=True)
    
    # Sidebar with preferences
    with st.sidebar:
        st.header("Preferences")
        
        weather = st.selectbox(
            "Weather",
            ['', 'Sunny/Hot', 'Cold', 'Rainy', 'Mild']
        )
        
        event = st.selectbox(
            "Event/Occasion",
            ['', 'Casual Outing', 'Work/Office', 'Date Night', 'Party', 'Gym', 'Formal Event', 'Beach', 'Brunch']
        )
        
        style_preference = st.multiselect(
            "Style Preferences",
            ['Barbiecore', 'Cottagecore', 'Dark Academia', 'Streetwear', 'Minimalist', 'Y2K', 'Coastal Grandmother', 'Clean Girl']
        )
        
        st.markdown("---")
        
        if st.button("Back to Wardrobe"):
            st.session_state.current_page = 'upload'
            st.rerun()
    
    # Main tabs
    tab1, tab2 = st.tabs(["Chat with AI", "Generate Outfits"])
    
    # Tab 1: Chat
    with tab1:
        st.subheader("Ask me anything about your style!")
        
        # Chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"<div class='chat-message user-message'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-message ai-message'><strong>AI:</strong> {message['content']}</div>", unsafe_allow_html=True)
        
        # Suggestion chips
        st.write("*Quick Questions:*")
        col1, col2, col3, col4 = st.columns(4)
        
        suggestions = [
            "What should I wear today?",
            "Show me Barbiecore outfits",
            "What's trending now?",
            "Outfit for a date night"
        ]
        
        for idx, suggestion in enumerate(suggestions):
            with [col1, col2, col3, col4][idx]:
                if st.button(suggestion, key=f"suggest_{idx}"):
                    st.session_state.pending_query = suggestion
        
        # Chat input
        user_query = st.chat_input("Ask about outfits, trends, or style advice...")
        
        if 'pending_query' in st.session_state:
            user_query = st.session_state.pending_query
            del st.session_state.pending_query
        
        if user_query:
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_query
            })
            
            # Analyze query
            query_analysis = understand_query(user_query)
            
            # Generate AI response
            ai_response = generate_ai_response(
                user_query,
                query_analysis,
                st.session_state.wardrobe_items
            )
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': ai_response
            })
            
            st.rerun()
    
    # Tab 2: Generate Outfits
    with tab2:
        st.subheader("Generate Perfect Outfits")
        
        query_input = st.text_input(
            "Describe the outfit you want:",
            placeholder="e.g., casual summer barbiecore outfit, formal business look, trendy streetwear"
        )
        
        if st.button("Generate Outfits", type="primary", use_container_width=True):
            if not query_input:
                st.warning("Please describe what kind of outfit you're looking for!")
            elif not st.session_state.wardrobe_items:
                st.error("Your wardrobe is empty! Please upload some items first.")
            else:
                with st.spinner("Creating perfect outfits for you..."):
                    # Ensure model is loaded
                    if not st.session_state.model_loaded:
                        model, preprocess, device = load_clip_model()
                        st.session_state.model = model
                        st.session_state.preprocess = preprocess
                        st.session_state.device = device
                        st.session_state.model_loaded = True
                    
                    # Analyze query
                    query_analysis = understand_query(query_input)
                    
                    # Show what AI understood
                    st.info(f"AI Understanding: Looking for *{query_analysis['occasion']}* outfits" + 
                           (f" with *{', '.join(query_analysis['trends'])}* style" if query_analysis['trends'] else ""))
                    
                    # Get preferences
                    preferences = {
                        'weather': weather if weather else None,
                        'event': event if event else None
                    }
                    
                    # Generate outfits
                    outfits, individual_items = generate_outfit_recommendations(
                        query_analysis,
                        st.session_state.wardrobe_items,
                        st.session_state.model,
                        st.session_state.device,
                        preferences
                    )
                    
                    if outfits:
                        st.success(f"Found {len(outfits)} perfect outfit combinations!")
                        
                        # Display outfits
                        for outfit_idx, outfit in enumerate(outfits):
                            match_percentage = int(outfit['avg_score'] * 100)
                            
                            with st.expander(f"Outfit #{outfit_idx + 1} - {match_percentage}% Match", expanded=(outfit_idx == 0)):
                                st.markdown(f"<div class='outfit-card'><div class='match-score'>{match_percentage}%</div><p>Perfect match for your style!</p></div>", unsafe_allow_html=True)
                                
                                cols = st.columns(len(outfit['items']))
                                for item_idx, item in enumerate(outfit['items']):
                                    with cols[item_idx]:
                                        img_data = base64.b64decode(item['image'])
                                        img = Image.open(io.BytesIO(img_data))
                                        st.image(img)
                                        st.caption(f"{item['category'].title()}")
                                        st.caption(f"Match: {int(item['match_score'] * 100)}%")
                                        if item.get('description'):
                                            st.caption(f"{item['description'][:50]}")
                        
                        # Show top individual items
                        st.markdown("---")
                        st.subheader("Top Matching Individual Items")
                        
                        cols = st.columns(5)
                        for idx, item in enumerate(individual_items[:5]):
                            with cols[idx]:
                                img_data = base64.b64decode(item['image'])
                                img = Image.open(io.BytesIO(img_data))
                                st.image(img)
                                st.caption(f"{item['category'].title()}")
                                st.caption(f"{int(item['match_score'] * 100)}%")
                    else:
                        st.warning("Couldn't create complete outfits. Try adding more items to your wardrobe!")

# Main app
def main():
    if st.session_state.current_page == 'upload':
        upload_page()
    else:
        assistant_page()

if __name__ == "__main__":
    main()
