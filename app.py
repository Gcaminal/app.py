# Importació llibreries
import streamlit as st
import requests
import datetime 
import pandas as pd
import matplotlib.pyplot as plt
try:
    from prophet import Prophet
except ImportError:
    st.error("No s'ha instal·lat 'prophet'. Executa a cmd: pip install prophet")
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ------------------------------------------------------------------------
# 1 CONFIGURACIÓ AIRTABLE
# ------------------------------------------------------------------------
AIRTABLE_PAT = "patDfe6ImZED5kfAE.efc27c8fe110443953d082a7e16aae214abbde770d3574c4fdf71d3cc10024ce"  # PAT complet
BASE_ID = "appIEZptaG5k4Auvh"               # ID de la base de dades

# IDs (tblXXXX) de cada taula Airtable 
COMANDA_TABLE = "tblydBjfNU9RNCVEl"
DETALL_TABLE = "tbllukBPzzo83xCe3"
INVENTARI_TABLE = "tbl4zHZASfatnnCNr"
CLIENT_TABLE = "tblpi3BYithjP2wI5"

# URLs per fer servir les API d'Airtable
comanda_url = f"https://api.airtable.com/v0/{BASE_ID}/{COMANDA_TABLE}"
detall_url = f"https://api.airtable.com/v0/{BASE_ID}/{DETALL_TABLE}"
inventari_url = f"https://api.airtable.com/v0/{BASE_ID}/{INVENTARI_TABLE}"
client_url = f"https://api.airtable.com/v0/{BASE_ID}/{CLIENT_TABLE}"

# Capçaleres API
headers = {
    "Authorization": f"Bearer {AIRTABLE_PAT}",
    "Content-Type": "application/json"
}

# ------------------------------------------------------------------------
# 2 FUNCIONS AUXILIARS (GET, CREATE, UPDATE)
# ------------------------------------------------------------------------

# Funció per llegir dades de la taula a la URL donada
def get_airtable_data(url):
    """
    Llegeix tots els registres de la taula a la URL donada (fent pàgines successives)
    i torna un DataFrame amb els seus camps + la columna 'record_id' interna d'Airtable.
    """
    all_records = []  # Aquí anirem acumulant tots els registres
    offset = None     # Parametre per la paginació
    while True:
        # Preparem els paràmetres de la crida GET. Si tenim offset, l'afegim.
        params = {}
        if offset:
            params["offset"] = offset

        # Fem la crida GET amb la pàgina corresponent
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            st.error(f"Error al obtenir dades (HTTP {response.status_code}): {response.text}")
            break

        # Convertim la resposta a JSON i recollim els registres
        data_json = response.json()
        records = data_json.get("records", [])

        # Afegim cada registre a la llista 'all_records'
        for r in records:
            fields = r.get("fields", {})
            fields["record_id"] = r["id"]  # Guardem la ID interna d'Airtable
            all_records.append(fields)

        # Mirem si Airtable ens ha retornat un 'offset' per a la pàgina següent
        offset = data_json.get("offset")

        # Si no hi ha 'offset', vol dir que ja hem arribat a l'última pàgina
        if not offset:
            break

    return pd.DataFrame(all_records)

    
# Funció per canviar IDs de camps amb link a una altra taula a valors llegibles
def map_linked_fields(dataframe, link_column, linked_table_url, key_column, value_column):
    """
    Reemplaça IDs a una columna vinculada (link_column) amb valors llegibles
    de la tabla vinculada especificada.
    """
    # Obtenir dades de la taula vinculada
    linked_table_data = get_airtable_data(linked_table_url)
    if not linked_table_data.empty:
        # Crear diccionari de ID a valors, aseguran que les claus siguinn cadenes
        id_to_value = {
            str(k): str(v) for k, v in linked_table_data.set_index(key_column)[value_column].to_dict().items()
        }
        
        # Funció per tobar IDs
        def map_ids(ids):
            if isinstance(ids, list):  # Cas: llista de IDs
                return ", ".join(id_to_value.get(str(id_), "Unknown") for id_ in ids)
            elif isinstance(ids, (str, int)):  # Cas: valor únic (str o int)
                return id_to_value.get(str(ids), "Unknown")
            elif pd.isna(ids):  # Cas: valor nul (NaN o None)
                return "Unknown"
            else:  # Altres
                return "Unknown"
        
        # Aplicar la funció de ratreig
        dataframe[link_column] = dataframe[link_column].apply(map_ids)
    else:
        st.warning(f"No s'ha pogut cargar la taula vinculada des de {linked_table_url}.")
    return dataframe

# Funció per crear un registre a la taula corresponent
def create_airtable_record(url, fields_dict):
    """
    Crea un registre a la taula corresponent fent servir un POST.
    fields_dict és un diccionari amb els camps, ex:
       {"OrderID": "123", "Status": "Valid", ...}
    """
    payload = {
        "records": [
            {
                "fields": fields_dict
            }
        ]
    }
    return requests.post(url, headers=headers, json=payload)

# Funció per actualitzar registres
def update_airtable_record(url, record_id, fields_dict):
    """
    Actualitza un registre a la taula corresponent fent servir PATCH.
    record_id és la ID interna "recXXX...".
    fields_dict és un diccionari amb els camps que cal actualitzar.
    """
    patch_url = f"{url}/{record_id}"
    payload = {
        "fields": fields_dict
    }
    return requests.patch(patch_url, headers=headers, json=payload)

# ------------------------------------------------------------------------
# 3 CONFIGURACIÓ DE PÀGINA A STREAMLIT
# ------------------------------------------------------------------------
st.set_page_config(
    page_title="Gestió de Comandes",
    page_icon="📦",
    layout="wide"
)

# Barra lateral
menu = st.sidebar.selectbox(
    "Navegació",
    ["Inici", "Comandes", "Detall comanda", "Inventari", "Client", "Anàlisi Predictiva"]
)

# ==========================
#      4 SECCIONS
# ==========================

# ---------- Pantalla d'Inici ----------
if menu == "Inici":
    st.title("Benvingut a la Gestió de Comandes")
    st.write("Aquesta aplicació t'ajuda a gestionar les teves comandes i l'estoc de forma eficient.")

# ---------- Comandes ----------
elif menu == "Comandes":
    st.header("Gestió de Comandes")

    
    # OBTENIR COMANDES (FILES EN ORDRE DE LA VISTA "Grid view")
    comandes_url = f"https://api.airtable.com/v0/{BASE_ID}/{COMANDA_TABLE}?view=Grid%20view"
    comandes_df = get_airtable_data(comandes_url)

    # Reordenar columnes 
    desired_order = ["OrderID", "CustomerID", "Status", "Data", "Detall comanda", "record_id"]
    cols_present = [c for c in desired_order if c in comandes_df.columns]
    comandes_df = comandes_df.reindex(columns=cols_present)

    if not comandes_df.empty:
        
        # Canviem el ID intern del client, i detall comanda pel ID que es mostra
        comandes_df = map_linked_fields(comandes_df, "CustomerID", client_url, "record_id", "CustomerID")
        comandes_df = map_linked_fields(comandes_df, "Detall comanda", detall_url, "record_id", "OrderID")

        st.subheader("Llistat de Comandes Existents")
        st.dataframe(comandes_df)

        # -------------------------------------------------------------------
        # 1) ACTUALIZAR "STATUS" DE UNA COMANDA EXISTENT
        # -------------------------------------------------------------------
        if "OrderID" in comandes_df.columns and "record_id" in comandes_df.columns:
            st.write("### Actualitzar l'estat d'una Comanda")

            selected_order = st.selectbox(
                "Selecciona una comanda:",
                comandes_df["OrderID"]
            )
            new_status = st.selectbox("Nou estat:", ["Valid", "Invalid", "Duplicate", "Pending"])

            if st.button("Actualitza Estat"):
                # Obtenim el 'record_id' de la fila que coincideix amb la comanda triada
                record_id = comandes_df.loc[
                    comandes_df["OrderID"] == selected_order, "record_id"
                ].values[0]

                # Cridar PATCH
                resp = update_airtable_record(
                    comandes_url.replace("?view=Grid%20view", ""),  # Treu la vista (Grid view)
                    record_id,
                    {"Status": new_status}
                )
                if resp.status_code == 200:
                    st.success(f"Comanda {selected_order} actualitzada a {new_status}.")
                else:
                    st.error(f"Error al actualizar: {resp.status_code} | {resp.text}")
        else:
            st.warning("No trobo 'OrderID' o 'record_id' a comandes_df.")
    else:
        st.warning("No hi ha comandes o s'ha produït un error en obtenir-les.")

    # -------------------------------------------------------------------
    # 2) CREAR UNA COMANDA NOVA
    # -------------------------------------------------------------------
    st.write("### Crear una Nova Comanda")

    # a) LLEGIR LA TAULA DE CLIENTS PEL SELECTBOX
    client_df = get_airtable_data(client_url)  # Ajusta al teu ID de taula 'Client'
    if not client_df.empty and "CustomerID" in client_df.columns and "record_id" in client_df.columns:
        # Construim diccionari: { "CUST001": "recXXXXXXXX", ... }
        client_dict = dict(zip(client_df["CustomerID"], client_df["record_id"]))
        # Selectbox perqué l'usuari vegi "CUSTxxx" 
        selected_customer_id = st.selectbox("CustomerID:", list(client_dict.keys()))
        # Internament: la record_id real del client 
        selected_customer_record = client_dict[selected_customer_id]
    else:
        st.warning("No trobo dades de 'CustomerID' en la taula Client. Usaré un valor vacío.")
        selected_customer_record = None

    # b) Status inicial
    new_status_val = st.selectbox("Status inicial:", ["Valid", "Invalid", "Duplicate", "Pending"])

    if st.button("Crear Comanda"):
        # Si "CustomerID" es un 'Link to another record' a Airtable, 
        # has de passar un array amb la record ID real del client:
        fields = { 
            "CustomerID": [selected_customer_record] if selected_customer_record else [],
            "Status": new_status_val
        }
        resp = create_airtable_record(comandes_url.replace("?view=Grid%20view", ""), fields)
        if resp.status_code in (200, 201):
            st.success("Comanda creada correctament!")
        else:
            st.error(f"Error al crear la comanda: {resp.status_code} | {resp.text}")

# ---------- Detall comanda ----------
elif menu == "Detall comanda":
    st.header("Gestió del Detall de les Comandes")

     # OBTENIR DETALL (FILES EN ORDRE DE LA VISTA "Grid view")
    detall_url_view = f"https://api.airtable.com/v0/{BASE_ID}/{DETALL_TABLE}?view=Grid%20view"
    detall_df = get_airtable_data(detall_url_view)

    # Reordenar columnes 
    desired_order = ["OrderID", "Comanda", "ProductID", "Quantity", "record_id"]
    cols_present = [c for c in desired_order if c in detall_df.columns]
    detall_df = detall_df.reindex(columns=cols_present)

    if not detall_df.empty:
        # Canviem el ID intern de comanda, i producte pel ID que es mostra
        detall_df = map_linked_fields(detall_df, "Comanda", comanda_url, "record_id", "OrderID")
        detall_df = map_linked_fields(detall_df, "ProductID", inventari_url, "record_id", "ProductID")

        st.subheader("Llistat de Detall comanda")
        st.dataframe(detall_df)
    else:
        st.info("No hi ha detalls o s'ha produït un error en obtenir-los.")

    st.write("### Afegir un nou Detall de Comanda")

    # -----------------------------------------------------------------------
    # OBTENIR LLISTA DE COMANDES (PER ENLLAÇAR) DESDE LA TAULA “Comanda”
    # -----------------------------------------------------------------------
    comanda_url_view = f"https://api.airtable.com/v0/{BASE_ID}/{COMANDA_TABLE}?view=Grid%20view"
    comanda_df = get_airtable_data(comanda_url_view)

    # Construim diccionari: (134: "recAbCdEf", 82: "recXyZ123"...)
    comanda_dict = {}
    if not comanda_df.empty and "OrderID" in comanda_df.columns and "record_id" in comanda_df.columns:
        comanda_dict = dict(zip(comanda_df["OrderID"], comanda_df["record_id"]))
    else:
        st.warning("No trobo 'OrderID' i/o 'record_id' a la taula Comanda. No es podrà enllaçar la comanda.")

    # El user escull l'"OrderID" (ej. 134), peró internament fem servir el seu 'record_id'
    list_of_orderids = sorted(list(comanda_dict.keys()))
    selected_orderid = st.selectbox("Comanda (OrderID) vinculat:", list_of_orderids)
    selected_comanda_recid = comanda_dict[selected_orderid] if selected_orderid in comanda_dict else None

    # -----------------------------------------------------------------------
    #  OBTENIR LLISTA DE PRODUCTES (PER “ProductID”) DESDE “Inventari”
    # -----------------------------------------------------------------------
    # Construim diccionari
    inventari_url_view = f"https://api.airtable.com/v0/{BASE_ID}/{INVENTARI_TABLE}?view=Grid%20view"
    inventari_df = get_airtable_data(inventari_url_view)

    product_dict = {}
    if not inventari_df.empty and "ProductID" in inventari_df.columns and "record_id" in inventari_df.columns:
        # Construim diccionari: {"PROD001": "recProdA1B2C3", "PROD002": "recProdD4E5F6", ...}
        product_dict = dict(zip(inventari_df["ProductID"], inventari_df["record_id"]))
    else:
        st.warning("No trobo 'ProductID' i/o 'record_id' a la taula Inventari.")

    # El user escull l'"ProductID" (ej. 134), peró internament fem servir el seu 'record_id'
    product_ids_sorted = sorted(list(product_dict.keys()))
    selected_productid = st.selectbox("ProductID:", product_ids_sorted)
    selected_product_recid = product_dict[selected_productid] if selected_productid in product_dict else None

    # -----------------------------------------------------------------------
    # QUANTITY
    # -----------------------------------------------------------------------
    new_quantity = st.number_input("Quantity:", min_value=0, step=1)

    # -----------------------------------------------------------------------
    # 1) CREAR EL NOU DETALL
    # -----------------------------------------------------------------------
    if st.button("Afegir Detall"):
        # “Comanda” és link -> array amb la record_id
        # “ProductID” també es link -> array amb la record_id
        # “OrderID” és un camp calculat => NO l'enviem
        # "Quantity" és un número => s'introdueix manualment
        if selected_comanda_recid and selected_product_recid:
            fields = {
                "Comanda": [selected_comanda_recid],
                "ProductID": [selected_product_recid],
                "Quantity": new_quantity
            }
            # Insertar a la taula base (sin ?view=)
            detall_base_url = f"https://api.airtable.com/v0/{BASE_ID}/{DETALL_TABLE}"
            resp = create_airtable_record(detall_base_url, fields)
            if resp.status_code in (200, 201):
                st.success("Detall creat correctament!")
            else:
                st.error(f"Error al crear el detall: {resp.status_code} | {resp.text}")
        else:
            st.warning("No s'ha pogut obtenir la record_id de la Comanda o del Producte.")


# ---------- Inventari ----------
elif menu == "Inventari":
    st.header("Gestió d'Estoc / Inventari")

    # OBTENIR INVENTARI (FILES EN ORDRE DE LA VISTA "Grid view")
    inventari_view_url = f"https://api.airtable.com/v0/{BASE_ID}/{INVENTARI_TABLE}?view=Grid%20view"
    inventari_df = get_airtable_data(inventari_view_url)

    # Reordenar columnes 
    desired_order = ["ProductID", "ProductName", "Stock", "ReorderLevel", "Reposition", "Detall comanda", "record_id"]
    cols_present = [c for c in desired_order if c in inventari_df.columns]
    inventari_df = inventari_df.reindex(columns=cols_present)

    if not inventari_df.empty:
        # Canviem el ID intern de comanda, i producte pel ID que es mostra
        inventari_df = map_linked_fields(inventari_df, "Detall comanda", detall_url, "record_id", "OrderID")

        st.subheader("Inventari Actual")
        st.dataframe(inventari_df)
        # -----------------------------------------
        # 1) ACTUALITZAR STOCK DE UN PRODUCTE EXISTENTE
        # -----------------------------------------
        if "ProductID" in inventari_df.columns and "record_id" in inventari_df.columns:
            st.write("### Actualitzar Stock d'un producte")

            product_selected = st.selectbox("Producte:", inventari_df["ProductID"])
            new_stock_val = st.number_input("Nou Stock:", min_value=0, step=1)

            if st.button("Actualitza Stock"):
                # Localitzem la record_id que coincideix amb el producte triat
                record_id = inventari_df.loc[
                    inventari_df["ProductID"] == product_selected,
                    "record_id"
                ].values[0]

                # IMPORTANT: URL base sense parámetres de vista
                inventari_base_url = f"https://api.airtable.com/v0/{BASE_ID}/{INVENTARI_TABLE}"
                update_url = f"{inventari_base_url}/{record_id}"

                # Enviem sol el camp 'Stock'
                payload = {"fields": {"Stock": new_stock_val}}
                resp = requests.patch(update_url, headers=headers, json=payload)

                if resp.status_code == 200:
                    st.success(f"S'ha actualitzat el stock de {product_selected} a {new_stock_val}.")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
        else:
            st.warning("No trobo 'ProductID' o 'record_id' en inventari.")

        # -----------------------------------------
        # 2) CREAR NOU PRODUCTE
        # -----------------------------------------
        st.write("### Afegir un Nou Producte")
        new_prod_id = st.text_input("Nou ProductID:", "")
        new_prod_name = st.text_input("Nom del Producte:", "")
        new_prod_stock = st.number_input("Stock inicial:", min_value=0, step=1)
        new_prod_reorder = st.number_input("ReorderLevel:", min_value=0, step=1)

        # 4) Botó de creació
        if st.button("Crear Producte"):
            # Montem el diccionari de camps
            fields = {
                "ProductID": new_prod_id,
                "ProductName": new_prod_name,
                "Stock": new_prod_stock,
                "ReorderLevel": new_prod_reorder
            }

            # Fer servir la mateixa inventari_base_url sense ?view
            inventari_base_url = f"https://api.airtable.com/v0/{BASE_ID}/{INVENTARI_TABLE}"
            r = create_airtable_record(inventari_base_url, fields)

            if r.status_code in (200, 201):
                st.success(f"Producte {new_prod_id} creat correctament!")
            else:
                st.error(f"Error al crear el producte: {r.status_code} | {r.text}")
    else:
        st.warning("No hi ha dades d'inventari o error al carregar-les.")

# ---------- Client ----------
elif menu == "Client":
    st.header("Gestió de Clients")

    #  OBTENIR CLIENTS (FILES EN ORDRE DE LA VISTA "Grid view")
    client_view_url = f"https://api.airtable.com/v0/{BASE_ID}/{CLIENT_TABLE}?view=Grid%20view"
    clients_df = get_airtable_data(client_view_url)

    # Reordenar columnes
    desired_order = ["CustomerID", "Name", "Email", "Phone", "Address", "Registration Date", "Comanda"]
    cols_present = [c for c in desired_order if c in clients_df.columns]
    clients_df = clients_df.reindex(columns=cols_present)
    
    if not clients_df.empty:
        # Canviem el ID intern de comanda, i producte pel ID que es mostra
        clients_df = map_linked_fields(clients_df, "Comanda", comanda_url, "record_id", "OrderID")

        st.subheader("Llistat de Clients")
        st.dataframe(clients_df)
    else:
        st.info("No hi ha dades de Client o s'ha produït un error en obtenir-les.")

    # -----------------------------------------
    # 1) CREAR NOU CLIENT
    # -----------------------------------------
    st.write("### Afegir un Nou Client")
    new_cust_id = st.text_input("CustomerID:", "")
    new_name = st.text_input("Name:", "")
    new_email = st.text_input("Email:", "")
    new_phone = st.text_input("Phone:", "")
    new_address = st.text_input("Address:", "")
    new_regdate = st.text_input("Registration Date:", "2025-12-25")  

    # 4) Botó de creació
    if st.button("Crear Client"):
        # Montem el diccionari de camps
        fields = {
            "CustomerID": new_cust_id,
            "Name": new_name,
            "Email": new_email,
            "Phone": new_phone,
            "Address": new_address,              
            "Registration Date": new_regdate     
        }

        # Fer sevir la URL base SENSE ?view=... per POST
        client_base_url = f"https://api.airtable.com/v0/{BASE_ID}/{CLIENT_TABLE}"
        r = create_airtable_record(client_base_url, fields)

        if r.status_code in (200, 201):
            st.success(f"Client {new_cust_id} creat correctament!")
        else:
            st.error(f"Error al crear el client: {r.status_code} | {r.text}")

# ---------- Anàlisi predictiva ----------

elif menu == "Anàlisi Predictiva":
    import pandas as pd
    import numpy as np
    import datetime
    from prophet import Prophet
    import os
    import time

    st.header("Anàlisi Predictiva")
    st.write("### Prediccions de demanda amb Prophet (mensual, amb reentrenament).")

    # -------------------------------------------------------------
    # 1) Carguem INVENTARI (para mapear record_id -> ProductID)
    # -------------------------------------------------------------
    inventari_df = get_airtable_data(inventari_url)
    recordid_to_name = {}
    if not inventari_df.empty and "record_id" in inventari_df.columns and "ProductID" in inventari_df.columns:
        recordid_to_name = dict(zip(inventari_df["record_id"], inventari_df["ProductID"]))
    else:
        st.warning("No trobo 'record_id' i/o 'ProductID' a INVENTARI. Potser no es podran mapear productes correctament.")

    # -------------------------------------------------------------
    # 2) Carguem la taula DETALL COMANDA amb dades reals
    # -------------------------------------------------------------
    df_detall = get_airtable_data(detall_url)
    if df_detall.empty:
        st.warning("No hi ha dades a 'Detall comanda'.")
        st.stop()

    needed_cols = {"ProductID", "Quantity", "Data"}
    if not needed_cols.issubset(df_detall.columns):
        st.warning(f"Falten columnes {needed_cols} a df_detall.")
        st.stop()

    # Convertir 'Data' a datetime
    try:
        df_detall["Data"] = pd.to_datetime(df_detall["Data"], errors="coerce")
    except:
        st.warning("No s'ha pogut convertir la columna 'Data' a datetime. Revisa el format.")

    # Mapeig record_id -> 'PRODxxx'
    def map_product(val):
        if isinstance(val, list) and len(val) > 0:
            rid = val[0]
            return recordid_to_name.get(rid, rid)
        return val

    df_detall["ProductID_str"] = df_detall["ProductID"].apply(map_product)

    # -------------------------------------------------------------
    # 3) Selector de producte
    # -------------------------------------------------------------
    product_list = sorted(df_detall["ProductID_str"].dropna().unique().tolist())
    selected_prod = st.selectbox("Selecciona el ProductID a analitzar:", product_list)

    # Filtrar df_detall al producte
    df_prod = df_detall[df_detall["ProductID_str"] == selected_prod].copy()

    # Agrupar la demanda diaria (sumar Quantity)
    df_prod_grouped = (
        df_prod
        .groupby(df_prod["Data"].dt.date)["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"Data": "ds", "Quantity": "y"})
    )
    df_prod_grouped["ds"] = pd.to_datetime(df_prod_grouped["ds"], errors="coerce")

    st.write(f"#### Dades reals diàries del producte **{selected_prod}**:")
    st.dataframe(df_prod_grouped)

    st.write("---")

    # -------------------------------------------------------------
    # 4) Seleccionar mes de 2025 a predir
    # -------------------------------------------------------------
    st.write("### Predicció mensual amb reentrenament")

    meses_2025 = {
        "Gener (Enero)": 1,
        "Febrer (Febrero)": 2,
        "Març (Marzo)": 3,
        "Abril": 4,
        "Maig": 5,
        "Juny": 6,
        "Juliol": 7,
        "Agost": 8,
        "Setembre": 9,
        "Octubre": 10,
        "Novembre": 11,
        "Desembre": 12
    }
    mes_select = st.selectbox("Mes de 2025 per predir:", list(meses_2025.keys()))
    mes_num = meses_2025[mes_select]

    # Checkbox per fer servir model d'entrenament
    use_retrained = st.checkbox("Usar el model reentrenat (si existe) en lloc d'entrenar amb dades fins la data", value=False)

    # Botó per predir
    if st.button("Predir el mes seleccionat"):
        # Determinem la data del més a predir
        start_of_month = datetime.date(2025, mes_num, 1)
        cutoff = pd.to_datetime(start_of_month)

        # Debug: veure que hi ha a session_state
        st.write("Session state keys:", list(st.session_state.keys()))
        if "model_retrained" in st.session_state:
            st.write("Existe un model_retrained en session_state. ✓")
        else:
            st.write("No hay model_retrained en session_state. ✗")

        # A) Triar: o entrenar de zero o fer servir model_retrained
        if (not use_retrained) or ("model_retrained" not in st.session_state):
            # Entrenar con datos < cutoff
            train_df = df_prod_grouped[df_prod_grouped["ds"] < cutoff].copy()
            if train_df.empty:
                st.warning("No hi ha dades anteriors al mes seleccionat.")
                st.stop()

            model = Prophet()
            model.fit(train_df)
            st.write("**Model entrenat des de zero** (fins a la data).")
        else:
            # Fer servir el model a session_state
            model = st.session_state["model_retrained"]
            st.write("**S'utilitza el model reentrenat** (session_state).")

        # B) Fer forecast ~30 dies del mes
        days_in_month = 30
        future_start = cutoff
        future_end = cutoff + datetime.timedelta(days=days_in_month)
        future_dates = pd.date_range(start=future_start, end=future_end, freq="D")
        future_df = pd.DataFrame({"ds": future_dates})

        forecast = model.predict(future_df)

        st.write(f"## Predicció per al mes de {mes_select} de 2025")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]])

        # Comparar amb dades reals
        real_mes = df_prod_grouped[
            (df_prod_grouped["ds"] >= cutoff) &
            (df_prod_grouped["ds"] <= cutoff + datetime.timedelta(days=days_in_month))
        ].copy()

        compare = pd.merge(
            real_mes[["ds","y"]],
            forecast[["ds","yhat"]],
            on="ds", how="inner"
        )
        if compare.empty:
            st.info("No hi ha dades reals per aquest mes (o no coincideixen dates).")
        else:
            compare["error"] = compare["y"] - compare["yhat"]
            compare["abs_error"] = compare["error"].abs()
            mae = compare["abs_error"].mean()
            rmse = (compare["error"]**2).mean()**0.5

            st.write("### Comparació amb dades reals d'aquest mes:")
            st.dataframe(compare)
            st.write(f"**MAE**: {mae:.2f}  |  **RMSE**: {rmse:.2f}")

        st.write("---")
        st.write("### Reentrenar el model amb les dades reals d'aquest mes")

        if st.button("Reentrenar amb dades reals del mes actual"):
            # 1) Tornem a obtenir train_df (ds < cutoff)
            train_df = df_prod_grouped[df_prod_grouped["ds"] < cutoff].copy()
            # 2) Afegim real_mes
            new_train = pd.concat([train_df, real_mes], ignore_index=True)
            new_train.drop_duplicates(subset=["ds"], keep="last", inplace=True)

            model2 = Prophet()
            model2.fit(new_train)

            st.success("S'ha reentrenat el model amb les dades reals d'aquest mes!")
            st.info("En la propera predicció, marca la casella 'Usar el model reentrenat' i es farà servir aquest.")

            # Guardem a session_state
            st.session_state["model_retrained"] = model2

    # ----------------------------------------------------------------
    # Bloc de CLASSIFICACIÓ automàtica de l’estat de les Comandes
    # ----------------------------------------------------------------
    st.write("---")
    st.write("### Classificació automàtica de l'estat de les Comandes")

    COMANDA_URL_BASE = f"https://api.airtable.com/v0/{BASE_ID}/{COMANDA_TABLE}"
    df_comanda = get_airtable_data(COMANDA_URL_BASE)

    needed_classif = {"CustomerID", "Data", "Detall comanda", "Status", "record_id"}
    if not df_comanda.empty and needed_classif.issubset(df_comanda.columns):

        def extract_first(val):
            if isinstance(val, list) and len(val) > 0:
                return val[0]
            return str(val)

        try:
            df_comanda["Data"] = pd.to_datetime(df_comanda["Data"])
        except:
            st.warning("No s'ha pogut convertir 'Data' a datetime en Comanda.")

        df_comanda["CustomerID_first"] = df_comanda["CustomerID"].apply(extract_first)

        if not df_detall.empty and "record_id" in df_detall.columns:
            df_detall_index = df_detall.set_index("record_id", drop=False)
            def get_total_quantity(list_of_recs):
                if not isinstance(list_of_recs, list):
                    return 0
                tot = 0
                for rid in list_of_recs:
                    if rid in df_detall_index.index:
                        row = df_detall_index.loc[rid]
                        q = row["Quantity"] if "Quantity" in row else 0
                        if isinstance(q, pd.Series):
                            q = q.sum()
                        tot += q
                return tot

            df_comanda["TotalQuantity"] = df_comanda["Detall comanda"].apply(get_total_quantity)
        else:
            df_comanda["TotalQuantity"] = 0

        df_labeled = df_comanda.dropna(subset=["Status"]).copy()
        df_labeled = df_labeled[df_labeled["Status"].isin(["Valid","Invalid","Duplicate"])]

        if df_labeled.empty:
            st.warning("No hi ha comandes amb Status = Valid/Invalid/Duplicate per entrenar el model.")
        else:
            try:
                df_labeled["CustomerID_num"] = df_labeled["CustomerID_first"].astype("category").cat.codes
            except Exception as e:
                st.error(f"Error convertint 'CustomerID' a category: {e}")
                df_labeled["CustomerID_num"] = 0

            df_labeled["DayOfWeek"] = df_labeled["Data"].dt.dayofweek.fillna(0)
            X = df_labeled[["CustomerID_num","DayOfWeek","TotalQuantity"]]
            y = df_labeled["Status"]

            if st.button("Entrenar, classificar i actualitzar Comandes noves"):
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import classification_report

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                clf = RandomForestClassifier(n_estimators=50, random_state=42)
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)
                st.write("#### Resultats en test:")
                st.json(report)
                st.write("Accuracy:", report["accuracy"])

                df_new = df_comanda[df_comanda["Status"].isin([None,"","Pending"])].copy()
                if not df_new.empty:
                    df_new["CustomerID_first"] = df_new["CustomerID"].apply(extract_first)
                    df_new["CustomerID_num"] = df_new["CustomerID_first"].astype("category").cat.codes
                    df_new["DayOfWeek"] = pd.to_datetime(df_new["Data"], errors="coerce").dt.dayofweek.fillna(0)

                    if not df_detall.empty and "record_id" in df_detall.columns:
                        df_detall_index2 = df_detall.set_index("record_id", drop=False)
                        def get_quantity2(list_of_recs):
                            if not isinstance(list_of_recs, list):
                                return 0
                            tot = 0
                            for rid in list_of_recs:
                                if rid in df_detall_index2.index:
                                    row = df_detall_index2.loc[rid]
                                    q = row["Quantity"] if "Quantity" in row else 0
                                    if isinstance(q, pd.Series):
                                        q = q.sum()
                                    tot += q
                            return tot
                        df_new["TotalQuantity"] = df_new["Detall comanda"].apply(get_quantity2)
                    else:
                        df_new["TotalQuantity"] = 0

                    X_new = df_new[["CustomerID_num","DayOfWeek","TotalQuantity"]].fillna(0)
                    y_pred_new = clf.predict(X_new)

                    st.write("Comandes noves classificades com:")
                    for idx, pred_label in zip(df_new.index, y_pred_new):
                        rec_id = df_new.loc[idx,"record_id"]
                        order_id_val = df_new.loc[idx,"OrderID"] if "OrderID" in df_new.columns else f"??_{idx}"

                        patch_url = f"{COMANDA_URL_BASE}/{rec_id}"
                        payload = {"fields": {"Status": pred_label}}
                        resp = requests.patch(patch_url, headers=headers, json=payload)
                        if resp.status_code == 200:
                            st.success(f"Comanda {order_id_val} -> {pred_label}")
                        else:
                            st.error(f"Error {resp.status_code}: {resp.text}")
                else:
                    st.info("No hi ha comandes noves en estat Pending.")
    else:
        st.warning(f"No hi ha dades a la taula Comanda o falten columnes {needed_classif}.")
