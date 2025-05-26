import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformer.app import BilingualTextHumanizer  # Updated import
import time
import io


def main():
    """
    Enhanced Streamlit app to humanize text using the bilingual humanizer:
    - Supports both English and French
    - Automatic language detection
    - Multiple tone options (academic/casual)
    - Advanced text analysis and visualization
    - Configurable parameters
    - Batch processing support
    """
    
    st.set_page_config(
        page_title="🌍 Bilingual AI Text Humanizer",
        page_icon="🧔🏻‍♂️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "# Advanced bilingual application to humanize AI-generated text in English and French with detailed analysis."
        }
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            margin-top: 0.5em;
            background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .intro {
            text-align: left;
            line-height: 1.8;
            margin-bottom: 1.5em;
            padding: 1em;
            background-color: #f0f2f6;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }
        .feature-box {
            background-color: #f8f9fa;
            padding: 1em;
            border-radius: 8px;
            margin: 0.5em 0;
            border: 1px solid #dee2e6;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1em;
            border-radius: 10px;
            text-align: center;
            margin: 0.5em;
        }
        .stButton > button {
            background: linear-gradient(90deg, #1f77b4, #ff7f0e);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 0.5em 2em;
            font-weight: bold;
        }
        .language-badge {
            display: inline-block;
            padding: 0.2em 0.8em;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            margin: 0.2em;
        }
        .lang-en {
            background-color: #e3f2fd;
            color: #1976d2;
            border: 1px solid #1976d2;
        }
        .lang-fr {
            background-color: #f3e5f5;
            color: #7b1fa2;
            border: 1px solid #7b1fa2;
        }
        .lang-auto {
            background-color: #e8f5e8;
            color: #388e3c;
            border: 1px solid #388e3c;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.markdown("<div class='title'>🌍 Bilingual AI Text Humanizer 🤖</div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class='intro'>
        <h4>🚀 Transform your AI text into natural, human-like content</h4>
        <p><b>🌍 Bilingual Features:</b></p>
        <ul>
        <li>🇺🇸🇫🇷 <b>Dual Language Support</b>: English and French processing</li>
        <li>🔍 <b>Smart Detection</b>: Automatic language identification</li>
        <li>🎯 <b>Multiple Tones</b>: Academic or casual styles</li>
        <li>🔄 <b>Intelligent Replacement</b>: Context-aware synonyms</li>
        <li>📊 <b>Detailed Analysis</b>: Complexity metrics and visualizations</li>
        <li>⚙️ <b>Configurable Parameters</b>: Precise transformation control</li>
        <li>📁 <b>Batch Processing</b>: Multiple files simultaneously</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for advanced configuration
    with st.sidebar:
        st.header("⚙️ Advanced Configuration")
        
        # Language selection
        language_option = st.selectbox(
            "🌍 Language Processing",
            options=["auto", "en", "fr"],
            format_func=lambda x: {
                "auto": "🔍 Auto-detect",
                "en": "🇺🇸 English",
                "fr": "🇫🇷 French"
            }[x],
            help="Choose language processing mode"
        )
        
        # Tone selection
        tone = st.selectbox(
            "🎨 Text Tone",
            options=["academic", "casual"],
            format_func=lambda x: "🎓 Academic" if x == "academic" else "💬 Casual",
            help="Choose the desired transformation style"
        )
        
        st.markdown("---")
        
        # Advanced parameters
        st.subheader("🔧 Transformation Parameters")
        
        p_synonym = st.slider(
            "Synonym Replacement",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Probability of replacing a word with a synonym"
        )
        
        p_restructure = st.slider(
            "Sentence Restructuring",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Probability of restructuring a sentence"
        )
        
        p_transition = st.slider(
            "Transition Addition",
            min_value=0.0,
            max_value=0.5,
            value=0.15,
            step=0.05,
            help="Probability of adding transition words"
        )
        
        p_errors = st.slider(
            "Natural Imperfections",
            min_value=0.0,
            max_value=0.2,
            value=0.05,
            step=0.01,
            help="Probability of adding subtle natural imperfections"
        )
        
        preserve_meaning = st.checkbox(
            "🔒 Preserve Strict Meaning",
            value=True,
            help="Use semantic similarity for better synonym selection"
        )
        
        show_analysis = st.checkbox(
            "📊 Show Detailed Analysis",
            value=True,
            help="Display metrics and analysis charts"
        )

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Input Text")
        
        # Text input methods
        input_method = st.radio(
            "Input Method:",
            ["Direct Input", "Single File", "Multiple Files"],
            horizontal=True
        )
        
        user_text = ""
        files_data = []
        
        if input_method == "Direct Input":
            user_text = st.text_area(
                "Enter your text here:",
                height=200,
                placeholder="Paste your AI-generated text here...\n\nSupports both English and French text!"
            )
            
        elif input_method == "Single File":
            uploaded_file = st.file_uploader(
                "Upload a file:",
                type=["txt", "md"],
                help="Supported formats: .txt, .md"
            )
            if uploaded_file is not None:
                user_text = uploaded_file.read().decode("utf-8", errors="ignore")
                st.text_area("File Preview:", value=user_text[:500] + "..." if len(user_text) > 500 else user_text, height=100)
                
        else:  # Multiple files
            uploaded_files = st.file_uploader(
                "Upload multiple files:",
                type=["txt", "md"],
                accept_multiple_files=True,
                help="Select multiple files for batch processing"
            )
            if uploaded_files:
                for file in uploaded_files:
                    content = file.read().decode("utf-8", errors="ignore")
                    files_data.append({"name": file.name, "content": content})
                st.success(f"✅ {len(files_data)} file(s) loaded")

    with col2:
        st.subheader("✨ Transformed Result")
        
        if st.button("🚀 Transform Text", type="primary"):
            if input_method != "Multiple Files" and not user_text.strip():
                st.warning("⚠️ Please enter or upload text to transform.")
            elif input_method == "Multiple Files" and not files_data:
                st.warning("⚠️ Please upload at least one file.")
            else:
                with st.spinner("🔄 Transformation in progress..."):
                    try:
                        # Initialize bilingual humanizer with custom parameters
                        humanizer = BilingualTextHumanizer(
                            p_synonym_replacement=p_synonym,
                            p_sentence_restructure=p_restructure,
                            p_transition_add=p_transition,
                            p_minor_errors=p_errors,
                            tone=tone,
                            language=language_option,
                            seed=42  # For reproducible results
                        )
                        
                        if input_method != "Multiple Files":
                            # Single text processing
                            process_single_text(user_text, humanizer, preserve_meaning, show_analysis)
                        else:
                            # Batch processing
                            process_multiple_files(files_data, humanizer, preserve_meaning, show_analysis)
                            
                    except Exception as e:
                        st.error(f"❌ Error during transformation: {str(e)}")
                        st.info("💡 Make sure SpaCy models are installed:\n- `python -m spacy download en_core_web_sm`\n- `python -m spacy download fr_core_news_sm`")


def process_single_text(text, humanizer, preserve_meaning, show_analysis):
    """Process a single text input"""
    start_time = time.time()
    
    # Detect language first
    detected_lang = humanizer.detect_language(text)
    
    # Display language detection result
    lang_labels = {"en": "🇺🇸 English", "fr": "🇫🇷 French"}
    lang_classes = {"en": "lang-en", "fr": "lang-fr"}
    
    st.markdown(f"""
    <div class="language-badge {lang_classes[detected_lang]}">
        Detected Language: {lang_labels[detected_lang]}
    </div>
    """, unsafe_allow_html=True)
    
    # Get original text analysis
    original_analysis = humanizer.analyze_text_features(text)
    
    # Transform text
    transformed = humanizer.humanize_text(text, preserve_meaning=preserve_meaning)
    
    # Get transformed text analysis
    transformed_analysis = humanizer.analyze_text_features(transformed)
    
    processing_time = time.time() - start_time
    
    # Display results
    st.markdown("### 📋 Transformed Text")
    st.markdown(f"```\n{transformed}\n```")
    
    # Download button
    st.download_button(
        label="💾 Download Result",
        data=transformed,
        file_name=f"humanized_text_{detected_lang}.txt",
        mime="text/plain"
    )
    
    # Statistics
    display_statistics(original_analysis, transformed_analysis, processing_time)
    
    # Detailed analysis
    if show_analysis:
        display_detailed_analysis(original_analysis, transformed_analysis)


def process_multiple_files(files_data, humanizer, preserve_meaning, show_analysis):
    """Process multiple files in batch"""
    st.markdown("### 📁 Batch Processing")
    
    results = []
    progress_bar = st.progress(0)
    
    for i, file_data in enumerate(files_data):
        # Detect language for each file
        detected_lang = humanizer.detect_language(file_data['content'])
        lang_labels = {"en": "🇺🇸 English", "fr": "🇫🇷 French"}
        
        st.markdown(f"**Processing: {file_data['name']}** ({lang_labels[detected_lang]})")
        
        # Transform text
        transformed = humanizer.humanize_text(file_data['content'], preserve_meaning=preserve_meaning)
        
        results.append({
            "filename": file_data['name'],
            "language": detected_lang,
            "original": file_data['content'],
            "transformed": transformed
        })
        
        # Show preview
        with st.expander(f"Preview: {file_data['name']} ({lang_labels[detected_lang]})"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original:**")
                st.text_area("", value=file_data['content'][:300] + "...", height=100, key=f"orig_{i}")
            with col2:
                st.markdown("**Transformed:**")
                st.text_area("", value=transformed[:300] + "...", height=100, key=f"trans_{i}")
        
        progress_bar.progress((i + 1) / len(files_data))
    
    # Create downloadable combined file
    combined_results = "\n\n" + "="*50 + "\n\n".join([
        f"FILE: {result['filename']} (Language: {result['language'].upper()})\n\n{result['transformed']}" 
        for result in results
    ])
    
    st.download_button(
        label="💾 Download All Results",
        data=combined_results,
        file_name="humanized_texts_batch.txt",
        mime="text/plain"
    )
    
    # Language distribution summary
    lang_counts = {}
    for result in results:
        lang_counts[result['language']] = lang_counts.get(result['language'], 0) + 1
    
    st.markdown("### 📊 Processing Summary")
    cols = st.columns(len(lang_counts))
    for i, (lang, count) in enumerate(lang_counts.items()):
        lang_labels = {"en": "🇺🇸 English", "fr": "🇫🇷 French"}
        with cols[i]:
            st.metric(lang_labels[lang], f"{count} files")
    
    st.success(f"✅ {len(results)} file(s) processed successfully!")


def display_statistics(original_analysis, transformed_analysis, processing_time):
    """Display comparison statistics"""
    st.markdown("### 📊 Comparative Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Words",
            transformed_analysis['word_count'],
            delta=transformed_analysis['word_count'] - original_analysis['word_count']
        )
    
    with col2:
        st.metric(
            "Sentences",
            transformed_analysis['sentence_count'],
            delta=transformed_analysis['sentence_count'] - original_analysis['sentence_count']
        )
    
    with col3:
        st.metric(
            "Lexical Diversity",
            f"{transformed_analysis['lexical_diversity']:.3f}",
            delta=f"{transformed_analysis['lexical_diversity'] - original_analysis['lexical_diversity']:.3f}"
        )
    
    with col4:
        st.metric(
            "Processing Time",
            f"{processing_time:.2f}s"
        )


def display_detailed_analysis(original_analysis, transformed_analysis):
    """Display detailed analysis with charts"""
    st.markdown("### 🔍 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Average Sentence Length")
        comparison_data = pd.DataFrame({
            'Version': ['Original', 'Transformed'],
            'Average Length': [
                original_analysis['avg_sentence_length'],
                transformed_analysis['avg_sentence_length']
            ]
        })
        
        fig = px.bar(
            comparison_data, 
            x='Version', 
            y='Average Length',
            color='Version',
            title="Sentence Length Comparison",
            color_discrete_map={'Original': '#ff7f0e', 'Transformed': '#1f77b4'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🏷️ Part-of-Speech Distribution")
        
        # Combine POS data
        all_pos = set(original_analysis['pos_distribution'].keys()) | set(transformed_analysis['pos_distribution'].keys())
        pos_data = []
        
        for pos in all_pos:
            pos_data.extend([
                {'POS': pos, 'Count': original_analysis['pos_distribution'].get(pos, 0), 'Version': 'Original'},
                {'POS': pos, 'Count': transformed_analysis['pos_distribution'].get(pos, 0), 'Version': 'Transformed'}
            ])
        
        pos_df = pd.DataFrame(pos_data)
        
        fig = px.bar(
            pos_df, 
            x='POS', 
            y='Count', 
            color='Version',
            barmode='group',
            title="Part-of-Speech Distribution",
            color_discrete_map={'Original': '#ff7f0e', 'Transformed': '#1f77b4'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Complexity and language info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🧠 Complexity Score")
        complexity_data = pd.DataFrame({
            'Metric': ['Complexity'],
            'Original': [original_analysis['complexity_score']],
            'Transformed': [transformed_analysis['complexity_score']]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Original', x=complexity_data['Metric'], y=complexity_data['Original'], marker_color='#ff7f0e'))
        fig.add_trace(go.Bar(name='Transformed', x=complexity_data['Metric'], y=complexity_data['Transformed'], marker_color='#1f77b4'))
        fig.update_layout(title="Text Complexity Comparison")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🌍 Language Information")
        
        # Display language detection info
        detected_lang = original_analysis.get('detected_language', 'unknown')
        lang_info = {
            'en': {'name': 'English', 'flag': '🇺🇸', 'family': 'Germanic'},
            'fr': {'name': 'French', 'flag': '🇫🇷', 'family': 'Romance'}
        }
        
        if detected_lang in lang_info:
            info = lang_info[detected_lang]
            st.markdown(f"""
            **Detected Language:** {info['flag']} {info['name']}  
            **Language Family:** {info['family']}  
            **Processing Model:** SpaCy {detected_lang}_core_web_sm
            """)
        
        # Additional metrics
        st.markdown("**Text Metrics:**")
        st.markdown(f"- Characters: {len(original_analysis.get('text', ''))}")
        st.markdown(f"- Average word length: {sum(len(word) for word in original_analysis.get('text', '').split()) / max(len(original_analysis.get('text', '').split()), 1):.1f}")


# Utility functions for enhanced features
def get_language_info(lang_code):
    """Get detailed language information"""
    info = {
        'en': {
            'name': 'English',
            'native_name': 'English',
            'flag': '🇺🇸',
            'family': 'Indo-European → Germanic → West Germanic',
            'speakers': '1.5 billion',
            'countries': 'US, UK, Canada, Australia, etc.'
        },
        'fr': {
            'name': 'French',
            'native_name': 'Français',
            'flag': '🇫🇷',
            'family': 'Indo-European → Romance → Gallo-Romance',
            'speakers': '280 million',
            'countries': 'France, Canada, Belgium, Switzerland, etc.'
        }
    }
    return info.get(lang_code, {})


if __name__ == "__main__":
    main()