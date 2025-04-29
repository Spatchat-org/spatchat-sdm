# ... [imports and Earth Engine auth remain unchanged]

def create_map(presence_points=None):
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer('OpenStreetMap').add_to(m)

    # Draw presence points
    if presence_points is not None:
        try:
            df = pd.read_csv(presence_points.name)
            if {'latitude', 'longitude'}.issubset(df.columns):
                points_layer = folium.FeatureGroup(name="🟦 Presence Points")
                for _, row in df.iterrows():
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=3,
                        color='blue',
                        fill=True,
                        fill_opacity=0.7
                    ).add_to(points_layer)
                points_layer.add_to(m)
        except Exception as e:
            print(f"⚠️ Error reading CSV: {e}")

    # Draw reprojected rasters
    wgs84_dir = "predictor_rasters/wgs84"
    if os.path.exists(wgs84_dir):
        for tif in os.listdir(wgs84_dir):
            if tif.endswith(".tif"):
                try:
                    path = os.path.join(wgs84_dir, tif)
                    with rasterio.open(path) as src:
                        print(f"🌍 Raster: {tif}, CRS: {src.crs}")
                        if src.count == 0:
                            print(f"⚠️ Skipping empty raster: {tif}")
                            continue
                        bounds = src.bounds
                        img = src.read(1)

                        # ✅ Normalize elevation/slope for visibility
                        if np.nanmin(img) != np.nanmax(img):
                            img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img))

                        raster_layer = folium.raster_layers.ImageOverlay(
                            image=img,
                            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                            opacity=0.4,
                            colormap=lambda x: (0, 1, 0, x),
                            name=f"🟨 {tif}"
                        )
                        raster_layer.add_to(m)
                except Exception as e:
                    print(f"⚠️ Error displaying raster {tif}: {e}")

    folium.LayerControl(collapsed=False).add_to(m)

    raw_html = m.get_root().render()
    safe_html = html_lib.escape(raw_html)
    iframe = f"""<iframe srcdoc="{safe_html}" style="width:100%; height:600px; border:none;"></iframe>"""
    return iframe

# --- Gradio App Layout (updated choices) ---

with gr.Blocks() as app:
    gr.Markdown("## 🧬 Spatchat-SDM: Global Species Distribution Modeling")

    map_output = gr.HTML(value=create_map(), label="🗺️ Live Preview")

    with gr.Row():
        uploader = gr.File(label="📥 Upload Presence Points (CSV)")
        upload_status = gr.Markdown()

    with gr.Row():
        layer_selector = gr.CheckboxGroup(
            label="🌎 Select Environmental Predictors",
            choices=[f"bio{i}" for i in range(1, 20)] + ["elevation", "slope", "aspect", "ndvi", "landcover"]
        )
        fetch_btn = gr.Button("📥 Fetch Predictors")
        fetch_status = gr.Markdown()

    with gr.Row():
        run_btn = gr.Button("🚀 Run SDM Model")
        run_status = gr.Markdown()

    with gr.Row():
        show_map_btn = gr.Button("🎯 Show Suitability Map")
        suitability_map_output = gr.HTML()

    # --- Actions ---

    uploader.change(fn=handle_upload, inputs=[uploader], outputs=[map_output, upload_status])
    fetch_btn.click(fn=fetch_predictors, inputs=[layer_selector], outputs=[fetch_status, layer_selector, map_output])
    run_btn.click(fn=run_model, outputs=[run_status])
    show_map_btn.click(fn=show_suitability_map, outputs=[suitability_map_output])

# --- Launch App ---

app.launch()
