import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
	page_title = "Belajar klasifikasi jeruk",
	page_icon = ":tangerine:"
)

model = joblib.load("model_belajar_jeruk.joblib")

diameter = st.slider("diameter",5.0,8.0,7.0)
berat = st.slider("berat",140.0, 200.0, 150.0)
tebal_kulit = st.slider("tebal kulit",0.5, 1.3, 0.7)
kadar_gula = st.slider("kadar gula",8.0, 14.0, 10.0)
asal_daerah = st.pills("asal daerah", ["Kalimantan", "Jawa Barat", "Jawa Tengah"], default="Kalimantan")
warna = st.pills("warna", ["hijau", "kuning", "oranye"], default="hijau")
musim_panen = st.pills("musim panen", ["kemarau", "hujan",], default="kemarau")


if st.button("Prediksi", type="primary"):
	data_baru = pd.DataFrame([[diameter, berat, tebal_kulit, kadar_gula, asal_daerah, warna, musim_panen]], 		columns=["diameter","berat","tebal_kulit","kadar_gula","asal_daerah","warna","musim_panen"])

	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"model meprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.caption("dibuat oleh al")

