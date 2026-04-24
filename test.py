import requests
import json

BASE_URL = "http://localhost:8000"


def print_response(label, response):
    print(f"\n{'─' * 50}")
    print(f"  {label}  [{response.status_code}]")
    print(f"{'─' * 50}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


def run_tests():

    # ─────────────────────────────────────────────────────
    #  GET /
    # ─────────────────────────────────────────────────────
    print_response("GET /", requests.get(f"{BASE_URL}/"))

    # ─────────────────────────────────────────────────────
    #  GET /health
    # ─────────────────────────────────────────────────────
    print_response("GET /health", requests.get(f"{BASE_URL}/health"))

    # ─────────────────────────────────────────────────────
    #  POST /predict — cas 1 : retour défectueux, client à risque
    # ─────────────────────────────────────────────────────
    cas1 = {
        "Customer_Gender": "Male",
        "Customer_Age": 35,
        "Customer_Wilaya": "Alger",
        "Customer_Past_Returns": 8,
        "Shop_Name": "TechnoStore",
        "Product_Category": "Electronics",
        "Product_Price_DA": 25000.0,
        "Order_Quantity": 1,
        "Total_Amount_DA": 25800.0,
        "Payment_Method": "CCP",
        "Shipping_Method": "Express",
        "Shipping_Cost_DA": 800.0,
        "Return_Reason": "Defective",
        "Days_to_Return": 3,
        "Shop_Return_Window_Days": 14,
        "Within_Return_Policy": 1,
        "Fraud_Score": 15.0,
        "Customer_Satisfaction": 2,
        "Is_Suspicious": 0,
    }
    print_response(
        "POST /predict — Défectueux / client à risque",
        requests.post(f"{BASE_URL}/predict", json=cas1),
    )

    # ─────────────────────────────────────────────────────
    #  POST /predict — cas 2 : mauvaise taille, retour tardif
    # ─────────────────────────────────────────────────────
    cas2 = {
        "Customer_Gender": "Female",
        "Customer_Age": 22,
        "Customer_Wilaya": "Oran",
        "Customer_Past_Returns": 1,
        "Shop_Name": "FashionHub",
        "Product_Category": "Clothing",
        "Product_Price_DA": 3500.0,
        "Order_Quantity": 2,
        "Total_Amount_DA": 7300.0,
        "Payment_Method": "Cash_on_Delivery",
        "Shipping_Method": "Standard",
        "Shipping_Cost_DA": 300.0,
        "Return_Reason": "Wrong_Size",
        "Days_to_Return": 12,
        "Shop_Return_Window_Days": 14,
        "Within_Return_Policy": 1,
        "Fraud_Score": 5.0,
        "Customer_Satisfaction": 3,
        "Is_Suspicious": 0,
    }
    print_response(
        "POST /predict — Mauvaise taille / retour tardif",
        requests.post(f"{BASE_URL}/predict", json=cas2),
    )

    # ─────────────────────────────────────────────────────
    #  POST /predict — cas 3 : fraude probable
    # ─────────────────────────────────────────────────────
    cas3 = {
        "Customer_Gender": "Male",
        "Customer_Age": 40,
        "Customer_Wilaya": "Constantine",
        "Customer_Past_Returns": 12,
        "Shop_Name": "MegaShop",
        "Product_Category": "Home_Appliances",
        "Product_Price_DA": 60000.0,
        "Order_Quantity": 3,
        "Total_Amount_DA": 61500.0,
        "Payment_Method": "CCP",
        "Shipping_Method": "Express",
        "Shipping_Cost_DA": 1500.0,
        "Return_Reason": "Not_As_Described",
        "Days_to_Return": 1,
        "Shop_Return_Window_Days": 30,
        "Within_Return_Policy": 1,
        "Fraud_Score": 95.0,
        "Customer_Satisfaction": 1,
        "Is_Suspicious": 1,
    }
    print_response(
        "POST /predict — Fraude probable",
        requests.post(f"{BASE_URL}/predict", json=cas3),
    )


if __name__ == "__main__":
    run_tests()