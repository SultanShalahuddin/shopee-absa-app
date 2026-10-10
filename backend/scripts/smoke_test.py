from fastapi.testclient import TestClient

from app.main import app


def main() -> None:
    with TestClient(app) as client:
        # Smoke test harus selalu terisolasi dari database yang dikonfigurasi
        # melalui .env. Integration test Supabase dijalankan secara terpisah dan
        # hanya dengan persetujuan eksplisit.
        client.app.state.database_service.enabled = False

        health = client.get("/health")
        assert health.status_code == 200, health.text
        demo_mode = health.json()["demo_mode"]
        assert health.json()["database_enabled"] is False
        if not demo_mode:
            assert health.json()["model_format"] == "onnx-int8"

        response = client.post(
            "/api/v1/predict",
            json={
                "review_text": (
                    "Aplikasinya cepat dan mudah digunakan, tetapi kurirnya lambat "
                    "dan voucher tidak bisa dipakai."
                )
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert len(payload["predictions"]) == 8
        assert payload["demo_mode"] is demo_mode
        assert payload["stored"] is False
        assert {item["aspect"] for item in payload["predictions"]} == {
            "iklan_dan_konten",
            "pengiriman_dan_kurir",
            "performa_aplikasi",
            "akun_login_keamanan",
            "pembayaran_cod_dan_keuangan",
            "promo_voucher_harga_dan_biaya",
            "penjual_dan_kualitas_produk",
            "retur_refund_dan_layanan",
        }

        history = client.get("/api/v1/history?limit=5")
        assert history.status_code == 200, history.text
        assert history.json() == {"items": []}

        if not demo_mode:
            predictions = {
                item["aspect"]: item["sentiment"] for item in payload["predictions"]
            }
            assert predictions["performa_aplikasi"] == "positif"
            assert predictions["pengiriman_dan_kurir"] == "negatif"

        mode = "demo" if demo_mode else "ONNX INT8 IndoBERT"
        print(f"Backend smoke test passed ({mode} mode)")


if __name__ == "__main__":
    main()
