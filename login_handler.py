import streamlit as st
import datetime
import time

# Session timeout in seconds (30 minutes)
SESSION_TIMEOUT = 30 * 60

def show_login_form():
    """Displays the login form and handles user authentication."""

    # Custom CSS for login page
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .login-title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        st.markdown('<h1 class="login-title">🔒 Smart Compliance Login</h1>', unsafe_allow_html=True)
        
        st.markdown("### Welcome to Smart Compliance Operations Unit Tool")
        st.markdown("Please enter your credentials to access the system.")

        # Using a form ensures that the page doesn't rerun on every widget interaction.
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username", placeholder="Enter your username")
            password = st.text_input(
                "Password", type="password", key="login_password", placeholder="Enter your password")

            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

            if submitted:
                if authenticate_user(username, password):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.session_state["login_time"] = time.time()  # Track login time
                    st.success("Login successful! Redirecting...")
                    # Use st.rerun() to immediately reflect the login state
                    st.rerun()
                else:
                    st.error("❌ Incorrect username or password")
        
        st.markdown('</div>', unsafe_allow_html=True)

def authenticate_user(username, password):
    """
    Authenticates the user against a dedicated 'credentials.users' list in secrets.toml.

    This is a more secure method as it specifically targets user credentials
    and avoids iterating over other sensitive keys.
    """
    try:
        # Access the specific list of users
        user_list = st.secrets["credentials"]["users"]

        # Find the user in the list
        for user in user_list:
            if user["username"] == username and user["password"] == password:
                return True
                
    except (KeyError, IndexError) as e:
        # Handle cases where secrets are not set up correctly
        st.error(f"Credential configuration error: {str(e)}. Please contact the administrator.")
        return False
    except Exception as e:
        st.error(f"Unexpected error during authentication: {str(e)}")
        return False

    return False

def add_logout_button():
    """Adds a logout button to the sidebar with user info."""
    # Display logged in user info
    if st.session_state.get("username"):
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"👤 **Logged in as:** {st.session_state.username}")
        
    if st.sidebar.button("🚪 Logout", type="secondary", use_container_width=True):
        # Clear the session state to log the user out
        st.session_state["authenticated"] = False
        st.session_state.pop("username", None)
        
        # Clear any other session data that should be reset on logout
        keys_to_clear = ['processing_results', 'last_processed_inputs', 
                        'progress_messages', 'show_persistent_results', 'current_page']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("You have been logged out successfully!")
        # Rerun the app to redirect to the login page
        st.rerun()

def check_session_timeout():
    """Check if the user session has timed out."""
    if not st.session_state.get("authenticated", False):
        return False
        
    login_time = st.session_state.get("login_time", 0)
    current_time = time.time()
    
    if current_time - login_time > SESSION_TIMEOUT:
        # Session timed out
        st.session_state["authenticated"] = False
        st.session_state.pop("username", None)
        st.session_state.pop("login_time", None)
        
        # Clear session data
        keys_to_clear = ['processing_results', 'last_processed_inputs', 
                        'progress_messages', 'show_persistent_results', 'current_page']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.warning("⏰ Your session has expired. Please log in again.")
        st.rerun()
        return False
    
    # Update last activity time
    st.session_state["login_time"] = current_time
    return True

def is_authenticated():
    """Check if user is authenticated and session is valid."""
    if not st.session_state.get("authenticated", False):
        return False
    
    return check_session_timeout()

def get_session_info():
    """Get information about the current session."""
    if not st.session_state.get("authenticated", False):
        return None
    
    login_time = st.session_state.get("login_time", 0)
    current_time = time.time()
    session_duration = current_time - login_time
    time_remaining = SESSION_TIMEOUT - session_duration
    
    return {
        "username": st.session_state.get("username", "Unknown"),
        "login_time": datetime.datetime.fromtimestamp(login_time),
        "session_duration": session_duration,
        "time_remaining": max(0, time_remaining)
    }
