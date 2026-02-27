import streamlit as st
import pandas as pd
import openai
from pypdf import PdfReader
import io
import os
import json
from datetime import datetime
import pytesseract
from pdf2image import convert_from_bytes

# --- Defined Variables ---
US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
]
DEFAULT_EXPORT_COLUMNS = [
    {"Header": "File Name", "Description": "Original file name of the invoice"},
    {"Header": "Invoice Number", "Description": "Invoice or document number"},
    {"Header": "Vendor", "Description": "Legal name of the vendor"},
    {"Header": "Ship From", "Description": "Origin location (City, State, County)"},
    {"Header": "Ship To", "Description": "Destination location (City, State, County)"},
    {"Header": "What is being sold", "Description": "Brief description of goods or services"},
    {"Header": "Total Amount", "Description": "Total invoice amount"},
    {"Header": "Tax Applied", "Description": "Sales tax amount charged"},
    {"Header": "Rate based on Total and Tax applied", "Description": "Effective tax rate calculated"},
    {"Header": "Rate for that ship to location", "Description": "Estimated combined tax rate"},
    {"Header": "Vendor History Note", "Description": "Whether vendor usually charges tax"}
]
# --- Setup Keys ---
import os

OPENAI_API_KEY_SECRET = os.getenv("OPENAI_API_KEY")
NGROK_TOKEN = os.getenv("NGROK_TOKEN")

if not OPENAI_API_KEY_SECRET:
    raise RuntimeError("KEY is not set")

# --- CONFIGURACIÓN DE PÁGINA ---

st.set_page_config(
  page_title="Taxlexia",
  layout="wide",
  page_icon="❇️",
  initial_sidebar_state="collapsed")

# --- INICIALIZACIÓN DE ESTADO ---

if 'vendor_history' not in st.session_state:
    st.session_state['vendor_history'] = pd.DataFrame(columns=[
        "Vendor", "Location", "Activity", "Has Charged Tax Before", "Last Seen"
    ])

if 'audit_results_df' not in st.session_state:
    st.session_state['audit_results_df'] = None

if "entities" not in st.session_state:
    st.session_state["entities"] = {}

if "active_entity" not in st.session_state:
    st.session_state["active_entity"] = None

# --- NUEVA FUNCIÓN DE EXTRACCIÓN INTELIGENTE (PDF + OCR) ---

def extract_text_from_pdf(file):
    text = ""
    filename = file.name

    # --- INTENTO 1: Lectura rápida de PDF nativo ---

    try:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\\n"

        # Si extrajimos una cantidad decente de texto, asumimos que funcionó
        # 100 caracteres es un umbral conservador para una factura
        if len(text.strip()) > 100:
            # st.toast(f"📄 {filename}: Read as native PDF.") # Opcional: feedback visual
            return text

        print(f"Info: Almost no text on {filename}. Using OCR...")

    except Exception as e:
        print(f"Advertencia: Almost no text on {filename}: {e}. Using OCR...")

    # --- INTENTO 2: OCR con Tesseract (Para imágenes/scans) ---
    try:
        # Importante: Resetear el puntero del archivo al inicio
        file.seek(0)
        pdf_bytes = file.read()

        # Convertir páginas del PDF a imágenes
        # Nota: En Colab esto usa poppler-utils instalado en el sistema
        images = convert_from_bytes(pdf_bytes, fmt='jpeg')

        ocr_text = ""
        # Barra de progreso interna para PDFs multipágina
        prog_text = st.empty()

        for i, image in enumerate(images):
            prog_text.caption(f"🔍 Processing page {i+1} of {len(images)} in {filename}...")
            # Aplicar OCR a la imagen. 'eng' optimiza para inglés.
            # Se puede usar 'eng+spa' si instalas el idioma español.
            page_text = pytesseract.image_to_string(image, lang='eng')
            ocr_text += f"--- Page {i+1} ---\\n{page_text}\\n"

        prog_text.empty()

        if ocr_text.strip():
            # st.toast(f"📷 {filename}: Read Successfully.") # Opcional
            return ocr_text
        else:
            return "Unable to read the file."

    except Exception as e_ocr:
        error_msg = f"Error Details: {e_ocr}"
        st.error(error_msg)
        return error_msg

# --- IA Analysis ---

def analyze_invoice_with_ai(text, user_location, business_type, export_columns):
    client = openai.OpenAI(api_key=OPENAI_API_KEY_SECRET)
    if text.startswith("Error"):
        return {"Error": text, "Vendor": "FILE_ERROR", "Total Amount": 0}

    fields_definition_list = []
    for h, d in zip(export_columns["Header"], export_columns["Description"]):
        fields_definition_list.append(f'- "{h}": {d}')
    fields_definition = "\\n".join(fields_definition_list)

    prompt = f"""
    Act as an expert Use and Sales Tax Auditor. 
    Analyze the following invoice text (which may contain OCR errors) and provide a STRICT JSON object using EXACTLY the below fields and the user context.
    Each key must match the Header exactly.
    
    User context (Buyer):
    - Fisical Location: {user_location}
    - Business Type: {business_type}

    Fields to extract:
    {fields_definition}

    Invoice text (Raw Text / OCR Output):
    {text[:5000]}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"Error": str(e), "Vendor": "AI_ERROR", "Total Amount": 0}

# --- PANTALLA DE LOGIN ---
def login_page(): # Renamed to avoid conflict
    st.markdown("""
    <style>
    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .feature-box {
        padding: 1.2rem;
        border-radius: 12px;
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        height: 100%;
    }
    .feature-title {
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- HEADER ---
    col_logo, col_login = st.columns([3,1])
    with col_logo:
        st.markdown("### ❇️ TaxlexIA")
    with col_login:
        st.markdown("")

    st.markdown(
        "<div class='main-title'>Automate your Use Tax determination.</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Extract vendor details, tax amounts, and activity logs from invoices instantly with AI.</div>",
        unsafe_allow_html=True
    )

    # --- BOTONES ---
    col_btn1, col_btn2 = st.columns([1,5])

    st.markdown("<br>", unsafe_allow_html=True)

    # --- FEATURES ---
    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
        <div class='feature-box'>
            <div class='feature-title'>Memory and Context</div>
            Persistent vendor, location, and transaction intelligence.
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class='feature-box'>
            <div class='feature-title'>Precise AI tax determination.</div>
            Jurisdiction-aware tax validation powered by AI.
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div class='feature-box'>
            <div class='feature-title'>Compliant and Back Up documentation</div>
            Export-ready audit trails and defensible records.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- LOGIN FORM ---
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "Barto" and password == "1234":
            st.session_state['authenticated'] = True
            st.session_state['username'] = username
            st.rerun()
        else:
            st.error("Invalid credentials")

# --- APP PRINCIPAL LOGIC ---
def main_app_logic():  # Renamed to main_app_logic to avoid conflict

    # 🚨 GUARD: no entities created yet
    if not st.session_state["entities"]:
        st.title("Welcome to TaxlexIA")
        st.info("🧩 Create an Entity to start analyzing invoices.")

        with st.form("create_first_entity"):
            new_entity_name = st.text_input("Entity name")
            submitted = st.form_submit_button("➕ Create Entity")

            if submitted and new_entity_name:
                st.session_state["entities"][new_entity_name] = {
                    "locations": [],
                    "vendors": pd.DataFrame(columns=[
                        "Vendor", "Location", "Activity", "Has Charged Tax Before", "Last Seen"
                    ]),
                    "export_columns": pd.DataFrame(DEFAULT_EXPORT_COLUMNS)
                }
                st.session_state["active_entity"] = new_entity_name
                st.rerun()

        return  # ⛔ no sigue ejecutando la app hasta que exista una entity

    # 👇 TODO lo que ya tenías sigue desde acá
    with st.sidebar:
        st.write(f"Welcome, {st.session_state['username']}")
        entity_names = list(st.session_state["entities"].keys())
        entity_names.append("➕ Add Entity")
        # 🔒 SAFETY: ensure active_entity is valid
        if st.session_state["active_entity"] not in entity_names:
            st.session_state["active_entity"] = entity_names[0]

        selected_entity = st.selectbox(
            "Active Entity",
            entity_names,
            index=entity_names.index(st.session_state["active_entity"])
        )
        if selected_entity == "➕ Add Entity":
            new_entity_name = st.text_input("New Entity Name")
            if st.button("Create Entity") and new_entity_name:
                st.session_state["entities"][new_entity_name] = {
                    "locations": [],
                    "vendors": pd.DataFrame(columns=[
                        "Vendor", "Location", "Activity", "Has Charged Tax Before", "Last Seen"
                    ]),
                    "export_columns": pd.DataFrame(DEFAULT_EXPORT_COLUMNS)
                }
                st.session_state["active_entity"] = new_entity_name
                st.rerun()
        else:
            st.session_state["active_entity"] = selected_entity

        st.divider()

        if st.button("Log out"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    entity = st.session_state["entities"][st.session_state["active_entity"]]

    tab1, tab2 = st.tabs(["⚙️ Account & Settings", "🗂️ Invoices Upload"])

    # --- TAB 1: SETTINGS ---
    with tab1:
        st.subheader("Physical Locations")
        for i, loc in enumerate(entity["locations"]):
            with st.container(border=True):
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)

                loc["state"] = col1.selectbox(
                    "State",
                    US_STATES,
                    index=US_STATES.index(loc.get("state", "CA")) if loc.get("state") in US_STATES else 0,
                    key=f"state_{i}"
                )

                loc["county"] = col2.text_input(
                    "County",
                    loc.get("county", ""),
                    key=f"county_{i}"
                )

                loc["city"] = col3.text_input(
                    "City",
                    loc.get("city", ""),
                    key=f"city_{i}"
                )

                loc["zip"] = col4.text_input(
                    "ZIP Code",
                    loc.get("zip", ""),
                    key=f"zip_{i}"
                )

                if st.button("🗑 Remove Location", key=f"remove_loc_{i}"):
                    entity["locations"].pop(i)
                    st.rerun()

        if st.button("➕ Add Physical Location"):
            entity["locations"].append({
                "state": "CA",
                "county": "",
                "city": "",
                "zip": ""
            })
            st.rerun()
        st.divider()

        st.subheader("Vendor History")
        entity["vendors"] = st.data_editor(
            entity["vendors"],
            num_rows="dynamic",
            use_container_width=True
        )

        st.divider()

        st.subheader("Export Columns")
        entity["export_columns"] = st.data_editor(
            entity["export_columns"],
            num_rows="dynamic",
            use_container_width=True
        )

        if st.button("Save Settings"):
            st.success("Saved")


    # --- TAB 2: BATCH UPLOAD ---
    with tab2:
        st.header("Invoices Upload")
        st.caption("ℹ️ This system supports PDF or Images. Note that image take longer to process.")

        uploaded_files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)

        if uploaded_files and st.button("🔍 Analyze Invoices"):
            user_loc = " | ".join(
                f'{l.get("city","")}, {l.get("state","")} {l.get("zip","")}'
                for l in entity["locations"]
                if l.get("state")
            ) or "Unknown"
            biz_type = st.session_state.get('user_prefs', {}).get('type', "General")

            results_list = []
            main_progress = st.progress(0)
            status_text = st.empty()

            for i, file in enumerate(uploaded_files):
                status_text.markdown(f"⏳ Processing **{file.name}** ({i+1}/{len(uploaded_files)})...")

                # 1. Extracción Inteligente (PDF o OCR)
                text = extract_text_from_pdf(file)

                # 2. Analisis
                data = analyze_invoice_with_ai(
                    text,
                    user_loc,
                    biz_type,
                    entity["export_columns"]
                )

                data['File Name'] = file.name
                results_list.append(data)

                # 3. Actualizar Historial (Solo si no hubo error crítico)
                if "Error" not in data or data.get("Vendor") != "FILE_ERROR":
                    has_tax = "Yes" if float(str(data.get("Tax Applied", 0)).replace('$','').replace(',','').strip() or 0) > 0 else "No"
                    new_vendor = {
                        "Vendor": data.get("Vendor", "Unknown"),
                        "Location": data.get("Ship From", "Unknown"),
                        "Activity": data.get("What is being sold", "Unknown"),
                        "HasChargedTaxBefore": has_tax,
                        "LastSeen": datetime.now().strftime("%Y-%m-%d")
                    }

                    entity["vendors"] = pd.concat(
                    [entity["vendors"], pd.DataFrame([new_vendor])],
                    ignore_index=True
                ).drop_duplicates(subset=["Vendor", "Location"], keep="last")

                main_progress.progress((i + 1) / len(uploaded_files))

            status_text.success("¡Análisis completado!")

            # 4. Preparar y guardar resultados
            df_results = pd.DataFrame(results_list)

            configured_headers = entity["export_columns"]["Header"].dropna().tolist()
            desired_columns = (
                configured_headers
                if configured_headers
                else df_results.columns.tolist())

            for col in desired_columns:
                if col not in df_results.columns:
                    df_results[col] = "N/A"

            st.session_state['audit_results_df'] = df_results[desired_columns]

        # --- SECCIÓN DE RESULTADOS Y DESCARGA ---
        if st.session_state['audit_results_df'] is not None:
            st.divider()
            st.subheader("Resultados del Análisis")
            st.dataframe(st.session_state['audit_results_df'], use_container_width=True)

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                st.session_state['audit_results_df'].to_excel(writer, sheet_name='Audit Results', index=False)
                worksheet = writer.sheets['Audit Results']
                for i, col in enumerate(st.session_state['audit_results_df'].columns):
                    worksheet.set_column(i, i, 20)

            st.download_button(
                label="📥 Download file (.xlsx)",
                data=buffer.getvalue(),
                file_name=f"Tax_Audit_OCR_{datetime.now().strftime('%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# --- MAIN STREAMLIT APP ENTRY POINT ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    login_page()
else:
    main_app_logic()
