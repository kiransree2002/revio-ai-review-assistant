"""
Employee authentication UI component.
Provides a modern login card, handles authentication against SQLite,
and manages secure session state.
"""

import streamlit as st
from src.database import authenticate_user


def render_login_page() -> None:
    """
    Renders the employee login gate.
    Validates credentials and saves session variables upon successful login.
    """
    st.markdown("""
    <div style="max-width: 480px; margin: 40px auto 16px auto; text-align: center;">
        <div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; border-radius: 14px; background: linear-gradient(135deg, #7C3AED, #38BDF8); font-size: 24px; box-shadow: 0 8px 20px rgba(124, 58, 237, 0.4); margin-bottom: 14px;">
            🔒
        </div>
        <h2 style="font-size: 2rem; font-weight: 800; color: #FFFFFF; margin: 0; letter-spacing: -0.02em;">
            Employee Portal Login
        </h2>
        <p style="color: #94A3B8; font-size: 0.95rem; margin: 8px 0 0 0;">
            Sign in with your authorized credentials to access your business intelligence dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_space1, col_form, col_space2 = st.columns([1, 1.8, 1])

    with col_form:
        st.markdown("""
        <style>
        .login-card {
            background: linear-gradient(145deg, rgba(20, 27, 45, 0.9) 0%, rgba(13, 18, 30, 0.9) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 28px 32px;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
        }
        </style>
        """, unsafe_allow_html=True)

        with st.form(key="employee_login_form"):
            st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#CBD5E1; margin-bottom:4px;'>User ID / Username</p>", unsafe_allow_html=True)
            username_input = st.text_input(
                "Username",
                placeholder="Enter your user ID",
                label_visibility="collapsed"
            )

            st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#CBD5E1; margin: 16px 0 4px 0;'>Password</p>", unsafe_allow_html=True)
            password_input = st.text_input(
                "Password",
                type="password",
                placeholder="••••••••••••",
                label_visibility="collapsed"
            )

            st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
            login_submitted = st.form_submit_button("Sign In to Dashboard", use_container_width=True)

            if login_submitted:
                if not username_input or not password_input:
                    st.error("Please enter both User ID and password.")
                else:
                    user_info = authenticate_user(username_input.strip(), password_input)
                    if user_info:
                        # Securely store authentication details in session state
                        st.session_state["authenticated"] = True
                        st.session_state["user_id"] = user_info["id"]
                        st.session_state["username"] = user_info["username"]
                        st.session_state["business_id"] = user_info["business_id"]
                        st.session_state["business_name"] = user_info["business_name"]
                        st.session_state["role"] = user_info["role"]
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please verify your user ID and password.")

        st.markdown("""
        <div style="margin-top: 18px; padding: 12px 14px; border-radius: 10px; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); text-align: center;">
            <span style="font-size: 0.78rem; color: #64748B;">
                Need to submit a review instead? 
                <a href="?page=submit_review&biz=demo-corp" target="_self" style="color: #38BDF8; font-weight: 600; text-decoration: none;">
                    Visit Public Review Form
                </a>
            </span>
        </div>
        """, unsafe_allow_html=True)


def logout_employee() -> None:
    """
    Clears all authentication state variables and reruns the app to the login screen.
    """
    auth_keys = [
        "authenticated",
        "user_id",
        "username",
        "business_id",
        "business_name",
        "role"
    ]
    for key in auth_keys:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

