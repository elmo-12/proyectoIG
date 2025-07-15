"""
Sistema de internacionalización para la aplicación
"""
import json
import os
import streamlit as st
from typing import Dict, Any

class I18n:
    """Clase para manejo de internacionalización"""
    
    def __init__(self):
        self.translations = {}
        self.current_language = 'es'
        self.supported_languages = {
            'es': {'name': 'Español', 'flag': '🇪🇸'},
            'en': {'name': 'English', 'flag': '🇺🇸'},
            'fr': {'name': 'Français', 'flag': '🇫🇷'},
            'pt': {'name': 'Português', 'flag': '🇧🇷'}
        }
        self.load_translations()
    
    def load_translations(self):
        """Cargar traducciones desde archivos JSON"""
        translations_dir = os.path.join(os.path.dirname(__file__), '../translations')
        
        for lang_code in self.supported_languages.keys():
            translation_file = os.path.join(translations_dir, f'{lang_code}.json')
            if os.path.exists(translation_file):
                with open(translation_file, 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
            else:
                self.translations[lang_code] = {}
    
    def set_language(self, lang_code: str):
        """Cambiar idioma actual"""
        if lang_code in self.supported_languages:
            self.current_language = lang_code
            st.session_state.language = lang_code
            # Los botones automáticamente refrescan la página
    
    def get_current_language(self) -> str:
        """Obtener idioma actual"""
        if 'language' in st.session_state:
            return st.session_state.language
        return self.current_language
    
    def t(self, key: str, **kwargs) -> str:
        """
        Traducir texto usando clave
        
        Args:
            key: Clave de traducción (ej: 'app.title')
            **kwargs: Variables para interpolación
            
        Returns:
            Texto traducido
        """
        current_lang = self.get_current_language()
        
        # Buscar traducción en idioma actual
        translation = self._get_nested_value(self.translations.get(current_lang, {}), key)
        
        # Si no existe, buscar en español como fallback
        if not translation and current_lang != 'es':
            translation = self._get_nested_value(self.translations.get('es', {}), key)
        
        # Si aún no existe, devolver la clave
        if not translation:
            translation = key
        
        # Interpolación de variables
        try:
            return translation.format(**kwargs)
        except (KeyError, ValueError):
            return translation
    
    def _get_nested_value(self, dictionary: Dict, key: str) -> Any:
        """Obtener valor anidado usando notación de punto"""
        keys = key.split('.')
        value = dictionary
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value
    
    def get_supported_languages(self) -> Dict[str, Dict[str, str]]:
        """Obtener idiomas soportados"""
        return self.supported_languages
    
    def render_language_selector(self):
        """Renderizar selector de idioma en la barra lateral"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🌐 Idioma / Language")
        
        # Crear dos filas de botones para los idiomas
        col1, col2 = st.sidebar.columns(2)
        
        # Primera fila: Español e Inglés
        with col1:
            if st.button("🇪🇸 Español", 
                        use_container_width=True, 
                        type="primary" if self.get_current_language() == "es" else "secondary"):
                self.set_language("es")
        
        with col2:
            if st.button("🇺🇸 English", 
                        use_container_width=True, 
                        type="primary" if self.get_current_language() == "en" else "secondary"):
                self.set_language("en")
        
        # Segunda fila: Francés y Portugués
        col3, col4 = st.sidebar.columns(2)
        
        with col3:
            if st.button("🇫🇷 Français", 
                        use_container_width=True, 
                        type="primary" if self.get_current_language() == "fr" else "secondary"):
                self.set_language("fr")
        
        with col4:
            if st.button("🇧🇷 Português", 
                        use_container_width=True, 
                        type="primary" if self.get_current_language() == "pt" else "secondary"):
                self.set_language("pt")

# Instancia global
i18n = I18n()

# Función de conveniencia
def t(key: str, **kwargs) -> str:
    """Función de conveniencia para traducir"""
    return i18n.t(key, **kwargs) 