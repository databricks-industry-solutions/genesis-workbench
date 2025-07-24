"""
Streamlit Usage Example: App Permissions Manager for Genesis Workbench

This example shows how to integrate the app permissions system with Streamlit
to control access to different modules and submodules based on user groups and access levels.
"""

try:
    import streamlit as st
except ImportError:
    # Streamlit not available in Databricks environment
    # This file is for reference only when deploying to external Streamlit environments
    raise ImportError(
        "Streamlit is not available in Databricks environment. "
        "This file is for external Streamlit deployments only."
    )

import os
from typing import List, Dict, Optional
from permissions_manager_app import (
    AppPermissionsManager,
    create_permissions_manager,
    get_user_accessible_modules,
    check_user_access,
)
from permissions_config import MODULES, ACCESS_LEVELS


def initialize_permissions_manager() -> AppPermissionsManager:
    """Initialize the permissions manager with environment variables."""
    return AppPermissionsManager(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN"),
    )


def get_current_user_groups(permissions_manager: AppPermissionsManager) -> List[str]:
    """Get the current user's groups from Databricks Identity API."""
    try:
        return permissions_manager.get_current_user_groups()
    except Exception as e:
        st.error(f"Error fetching user groups: {e}")
        return []


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_accessible_modules(user_groups: List[str]) -> Dict[str, Dict[str, str]]:
    """Get accessible modules with access levels, cached for better performance."""
    if not user_groups:
        return {}

    try:
        permissions_manager = initialize_permissions_manager()
        return permissions_manager.get_accessible_modules(user_groups)
    except Exception as e:
        st.error(f"Error getting accessible modules: {e}")
        return {}


def check_module_access(
    user_groups: List[str],
    module_name: str,
    submodule_name: Optional[str] = None,
    required_access_level: str = "view",
) -> bool:
    """Check if user has access to a specific module or submodule at the required level."""
    try:
        permissions_manager = initialize_permissions_manager()
        return permissions_manager.check_user_module_access(
            user_groups, module_name, submodule_name, required_access_level
        )
    except Exception as e:
        st.error(f"Error checking module access: {e}")
        return False


def create_navigation_sidebar(
    accessible_modules: Dict[str, Dict[str, str]],
) -> Optional[str]:
    """Create a navigation sidebar with accessible modules and their access levels."""
    st.sidebar.title("Genesis Workbench")

    if not accessible_modules:
        st.sidebar.warning("No accessible modules found.")
        return None

    st.sidebar.markdown("### Available Modules")

    # Create navigation options with access level indicators
    nav_options = []
    for module_name, submodules_dict in accessible_modules.items():
        module_config = MODULES.get(module_name)
        if module_config:
            module_access = submodules_dict.get("_module_access", "view")
            access_icon = "🔓" if module_access == "full" else "👁️"
            nav_options.append(f"{access_icon} {module_config.display_name}")

            for submodule, access_level in submodules_dict.items():
                if submodule != "_module_access":  # Skip the special module access key
                    sub_icon = "🔓" if access_level == "full" else "👁️"
                    nav_options.append(f"  └ {sub_icon} {submodule}")

    if nav_options:
        selected = st.sidebar.selectbox("Navigate to:", nav_options)
        return selected

    return None


def main_streamlit_app():
    """Main Streamlit application with permissions-based navigation."""
    st.set_page_config(page_title="Genesis Workbench", page_icon="🧬", layout="wide")

    # Initialize permissions manager
    try:
        permissions_manager = initialize_permissions_manager()
    except Exception as e:
        st.error(f"Failed to initialize permissions manager: {e}")
        st.stop()

    # Get current user groups
    with st.spinner("Loading user permissions..."):
        user_groups = get_current_user_groups(permissions_manager)

    if not user_groups:
        st.warning("No group memberships found. Contact your administrator.")
        st.stop()

    # Display user info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Groups:** {', '.join(user_groups)}")

    # Get accessible modules
    accessible_modules = get_cached_accessible_modules(user_groups)

    # Create navigation
    selected_nav = create_navigation_sidebar(accessible_modules)

    # Main content area
    st.title("🧬 Genesis Workbench")

    if not accessible_modules:
        st.error(
            "You don't have access to any modules. Please contact your administrator."
        )
        return

    # Display module content based on selection
    if selected_nav:
        if selected_nav.startswith("🔓") or selected_nav.startswith("👁️"):
            # Module selected
            module_display_name = selected_nav[2:].strip()  # Remove emoji and space
            module_name = None

            # Find module by display name
            for name, config in MODULES.items():
                if (
                    config.display_name == module_display_name
                    and name in accessible_modules
                ):
                    module_name = name
                    break

            if module_name:
                module_access_level = accessible_modules[module_name].get(
                    "_module_access", "view"
                )
                display_module_overview(
                    module_name, accessible_modules[module_name], module_access_level
                )
        else:
            # Submodule selected
            submodule_line = selected_nav.strip()
            if "└" in submodule_line:
                submodule_name = (
                    submodule_line.split("└")[1].strip()[2:].strip()
                )  # Remove tree char and emoji
                # Find which module this submodule belongs to
                for module_name, submodules_dict in accessible_modules.items():
                    if (
                        submodule_name in submodules_dict
                        and submodule_name != "_module_access"
                    ):
                        access_level = submodules_dict[submodule_name]
                        display_submodule_content(
                            module_name, submodule_name, access_level
                        )
                        break
    else:
        # Default dashboard
        display_dashboard(accessible_modules)


def display_dashboard(accessible_modules: Dict[str, Dict[str, str]]):
    """Display the main dashboard with module overview."""
    st.header("Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accessible Modules")
        for module_name, submodules_dict in accessible_modules.items():
            module_config = MODULES.get(module_name)
            if module_config:
                module_access = submodules_dict.get("_module_access", "view")
                access_badge = "🔓 Full" if module_access == "full" else "👁️ View"

                with st.expander(f"📋 {module_config.display_name} ({access_badge})"):
                    st.write(module_config.description)
                    if len(submodules_dict) > 1:  # More than just _module_access
                        st.write("**Available submodules:**")
                        for submodule, access_level in submodules_dict.items():
                            if submodule != "_module_access":
                                sub_badge = "🔓" if access_level == "full" else "👁️"
                                st.write(f"• {sub_badge} {submodule} ({access_level})")

    with col2:
        st.subheader("Quick Actions")
        st.info("Select a module from the sidebar to get started.")

        # Show some statistics
        total_modules = len(accessible_modules)
        total_submodules = sum(
            len(subs) - 1 for subs in accessible_modules.values()
        )  # -1 for _module_access
        full_access_count = sum(
            1
            for subs in accessible_modules.values()
            if subs.get("_module_access") == "full"
        )

        st.metric("Accessible Modules", total_modules)
        st.metric("Accessible Submodules", total_submodules)
        st.metric("Full Access Modules", full_access_count)


def display_module_overview(
    module_name: str, submodules_dict: Dict[str, str], module_access_level: str
):
    """Display overview for a specific module."""
    module_config = MODULES.get(module_name)
    if not module_config:
        st.error(f"Module configuration not found: {module_name}")
        return

    access_badge = "🔓 Full Access" if module_access_level == "full" else "👁️ View Only"
    st.header(f"📋 {module_config.display_name}")
    st.caption(f"Your access level: {access_badge}")
    st.write(module_config.description)

    # Filter out the special _module_access key
    submodules = {k: v for k, v in submodules_dict.items() if k != "_module_access"}

    if submodules:
        st.subheader("Available Submodules")

        cols = st.columns(min(len(submodules), 3))
        for i, (submodule, access_level) in enumerate(submodules.items()):
            with cols[i % 3]:
                access_icon = "🔓" if access_level == "full" else "👁️"
                if st.button(
                    f"{access_icon} {submodule}", key=f"sub_{module_name}_{submodule}"
                ):
                    # In a real app, this would navigate to the submodule
                    st.success(f"Opening {submodule} with {access_level} access...")
    else:
        st.warning("No submodules accessible in this module.")


def display_submodule_content(module_name: str, submodule_name: str, access_level: str):
    """Display content for a specific submodule with access level awareness."""
    module_config = MODULES.get(module_name)
    if not module_config:
        st.error(f"Module configuration not found: {module_name}")
        return

    access_badge = "🔓 Full Access" if access_level == "full" else "👁️ View Only"
    st.header(f"🔧 {submodule_name}")
    st.caption(f"Part of {module_config.display_name} • {access_badge}")

    # Show different content based on access level
    if access_level == "view":
        st.info("You have view-only access to this submodule.")
    else:
        st.success("You have full access to this submodule.")

    # Example content based on submodule type
    if submodule_name == "settings":
        st.subheader("Settings")
        st.write("Configure module-specific settings here.")

        if access_level == "full":
            # Full access users can modify settings
            with st.form("settings_form"):
                setting1 = st.text_input("Setting 1", value="default_value")
                setting2 = st.selectbox(
                    "Setting 2", ["Option A", "Option B", "Option C"]
                )
                submitted = st.form_submit_button("Save Settings")

                if submitted:
                    st.success("Settings saved successfully!")
        else:
            # View-only users see read-only settings
            st.text_input("Setting 1", value="default_value", disabled=True)
            st.selectbox(
                "Setting 2", ["Option A", "Option B", "Option C"], disabled=True
            )
            st.info("Contact your administrator to modify these settings.")

    elif "prediction" in submodule_name:
        st.subheader("Prediction Interface")
        st.write("Run predictions using trained models.")

        if access_level == "full":
            # Full access users can run predictions
            uploaded_file = st.file_uploader("Upload data for prediction")
            if uploaded_file is not None:
                if st.button("Run Prediction"):
                    st.success("Prediction completed successfully!")
        else:
            # View-only users can see results but not run new predictions
            st.info("You can view existing predictions but cannot run new ones.")
            st.text("Previous prediction results would be shown here...")

    elif "design" in submodule_name:
        st.subheader("Design Interface")
        st.write("Design new molecules or proteins.")

        if access_level == "full":
            # Full access users can create designs
            design_params = st.slider("Design Parameter", 0.0, 1.0, 0.5)
            if st.button("Generate Design"):
                st.success(f"Design generated with parameter: {design_params}")
        else:
            # View-only users can see existing designs
            st.info("You can view existing designs but cannot create new ones.")
            st.text("Existing designs would be displayed here...")

    else:
        st.subheader("Module Interface")
        st.write(f"Interface for {submodule_name} functionality.")

        if access_level == "view":
            st.info("You have read-only access to this functionality.")


def admin_permissions_interface():
    """Admin interface for managing permissions with access levels."""
    st.title("🔐 Permissions Management")

    # This would only be accessible to admin users
    permissions_manager = initialize_permissions_manager()
    user_groups = get_current_user_groups(permissions_manager)

    # Check if user is admin (has master_settings access)
    is_admin = check_module_access(user_groups, "master_settings")

    if not is_admin:
        st.error("Access denied. Admin privileges required.")
        return

    st.subheader("Grant Module Access")

    with st.form("grant_access_form"):
        col1, col2 = st.columns(2)

        with col1:
            module = st.selectbox("Module", list(MODULES.keys()))
            submodule = st.selectbox(
                "Submodule (optional)",
                [""] + MODULES[module].submodules if module in MODULES else [""],
            )

        with col2:
            access_level = st.selectbox("Access Level", list(ACCESS_LEVELS.keys()))
            groups = st.text_input(
                "Groups (comma-separated)", placeholder="group1,group2"
            )

        st.markdown(f"**Selected Access Level:** {ACCESS_LEVELS.get(access_level, '')}")

        if st.form_submit_button("Grant Access"):
            if groups:
                try:
                    group_list = [g.strip() for g in groups.split(",")]
                    permissions_manager.grant_module_access(
                        module_name=module,
                        groups=group_list,
                        access_level=access_level,
                        submodule_name=submodule if submodule else None,
                    )
                    st.success(f"Access granted successfully!")
                except Exception as e:
                    st.error(f"Error granting access: {e}")

    st.subheader("Update Access Level")

    with st.form("update_access_form"):
        col1, col2 = st.columns(2)

        with col1:
            update_module = st.selectbox(
                "Module", list(MODULES.keys()), key="update_module"
            )
            update_submodule = st.selectbox(
                "Submodule (optional)",
                (
                    [""] + MODULES[update_module].submodules
                    if update_module in MODULES
                    else [""]
                ),
                key="update_submodule",
            )

        with col2:
            update_group = st.text_input("Group Name", key="update_group")
            new_access_level = st.selectbox(
                "New Access Level", list(ACCESS_LEVELS.keys()), key="new_access"
            )

        if st.form_submit_button("Update Access Level"):
            if update_group:
                try:
                    permissions_manager.update_access_level(
                        module_name=update_module,
                        group_name=update_group,
                        new_access_level=new_access_level,
                        submodule_name=update_submodule if update_submodule else None,
                    )
                    st.success(f"Access level updated successfully!")
                except Exception as e:
                    st.error(f"Error updating access level: {e}")


if __name__ == "__main__":
    # Run the main app
    main_streamlit_app()

    # Add admin interface in sidebar if user has access
    try:
        permissions_manager = initialize_permissions_manager()
        user_groups = get_current_user_groups(permissions_manager)

        if check_module_access(user_groups, "master_settings"):
            st.sidebar.markdown("---")
            if st.sidebar.button("⚙️ Admin Panel"):
                admin_permissions_interface()
    except:
        pass  # Silently fail if permissions check fails
