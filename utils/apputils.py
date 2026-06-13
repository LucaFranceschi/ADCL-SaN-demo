root_css = """
.equal-radio .wrap {
    display: flex !important;
    flex-wrap: wrap;
}
.equal-radio .wrap label {
    flex: 1 1 0 !important;   /* all items grow equally */
}
"""

html_output_table = """
    <style>
    .comparison-table,
    .comparison-table *,
    .comparison-table tr,
    .comparison-table td,
    .comparison-table th,
    .comparison-table tbody,
    .comparison-table thead {{
        border: 0 !important;
        outline: none !important;
        box-shadow: none !important;
    }}

    .comparison-table {{
        width: 100% !important;
        table-layout: fixed !important;
        border-collapse: collapse !important;
        border-spacing: 0 !important;
        font-family: var(--font, ui-sans-serif, system-ui, sans-serif) !important;
        color: var(--body-text-color) !important;
    }}

    /* HARD FORCE COLUMN WIDTHS */
    .comparison-table col.label-col {{
        width: 24px !important;
    }}

    .comparison-table th,
    .comparison-table td {{
        padding: 2px !important;
        text-align: center !important;
        vertical-align: middle !important;
        border: none !important;
        line-height: 0 !important;
    }}
    .comparison-table th {{
        font-weight: 600 !important;
        font-size: 1rem !important;
        line-height: normal !important;
        padding-bottom: 6px !important;
    }}

    /* tiny first column */
    .comparison-table td.row-label,
    .comparison-table th.row-label {{
        width: 24px !important;
        min-width: 24px !important;
        max-width: 24px !important;
        padding: 0 !important;
        overflow: visible !important;
    }}

    .row-label-inner {{
        writing-mode: vertical-rl;
        transform: rotate(180deg);

        font-weight: 600;
        white-space: nowrap;

        width: 24px;
        margin: 0 auto;
    }}

    .img-container {{
        width: 100% !important;
        display: block !important;
        overflow: hidden !important;
        border-radius: 12px !important;
        font-size: 0 !important;
    }}
    .img-container img {{
        width: 100% !important;
        height: auto !important;
        max-width: 100% !important;
        max-height: 300px !important;
        object-fit: fill !important;
        display: block !important;
        aspect-ratio: 1 / 1 !important;
    }}
    </style>

    <table class="comparison-table">

        <colgroup>
            <col class="label-col">
            {}
        </colgroup>

        <thead>
            <tr>
                <th class="row-label"></th>
    """

html_empty_box_for_output = '<div class="block svelte-11xb1hd auto-margin" dir="ltr" style="height: 350px; border-style: solid; overflow: hidden; min-width: min(160px, 100%); border-width: var(--block-border-width) !important;"><div class="wrap default full svelte-ls20lj hide" style="position: absolute; padding: 0px;"></div> <label for="" data-testid="block-label" dir="ltr" class="svelte-1to105q float"><span class="svelte-1to105q"><svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="feather feather-image"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg></span> Output of all models</label> <div class="empty svelte-1oiin9d large unpadded_box" aria-label="Empty value"><div class="icon svelte-1oiin9d"><svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="feather feather-image"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg></div></div> </div>'

def images_to_html(images, col_labels, row_labels):
    html = html_output_table.format("".join("<col>" for _ in range(len(col_labels))))

    for label in col_labels:
        html += f"<th>{label}</th>"

    html += "</tr></thead><tbody>"

    for row_idx, row_label in enumerate(row_labels):
        html += f"""
        <tr>
            <td class="row-label">
                <div class="row-label-inner">{row_label}</div>
            </td>
        """

        for col_idx in range(len(col_labels)):
            img_src = images[col_idx * len(row_labels) + row_idx]

            html += f"""
            <td>
                <div class="img-container">
                    <img src="{img_src}">
                </div>
            </td>
            """

        html += "</tr>"

    html += "</tbody></table>"

    return html