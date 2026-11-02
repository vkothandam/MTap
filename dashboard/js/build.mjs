// Bundles src/space3d.js (the scene + three.js) into ../forecast_dash/space3d/space3d.js, the file
// Streamlit serves for the 3-D view (pyproject.toml [tool.streamlit.component], asset_dir).
import { build } from "esbuild";

await build({
  entryPoints: ["src/space3d.js"],
  outfile: "../forecast_dash/space3d/space3d.js",
  bundle: true,
  format: "esm",
  minify: true,
  target: "es2020",
  legalComments: "eof",   // three.js's MIT notice
});
