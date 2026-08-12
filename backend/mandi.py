"""
AgriSathi Mandi Price & Market Recommendation Engine
Tracks district-wise mandi prices (Agmarknet benchmark rates), 7-day price trends, and calculates transport-adjusted net profit recommendations.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Agmarknet Mandi Price Matrix (Rates in INR per Quintal)
MANDI_DATABASE = [
    # Wheat
    {"commodity": "Wheat", "state": "Punjab", "district": "Ludhiana", "mandi": "Khanna Mandi", "modal_price": 2285.0, "min_price": 2250.0, "max_price": 2310.0, "trend": "RISING", "trend_pct": 2.4, "distance_km": 35, "msp": 2275.0},
    {"commodity": "Wheat", "state": "Punjab", "district": "Ludhiana", "mandi": "Ludhiana Main Mandi", "modal_price": 2250.0, "min_price": 2220.0, "max_price": 2270.0, "trend": "STABLE", "trend_pct": 0.2, "distance_km": 12, "msp": 2275.0},
    {"commodity": "Wheat", "state": "Punjab", "district": "Jalandhar", "mandi": "Jalandhar Grain Market", "modal_price": 2260.0, "min_price": 2230.0, "max_price": 2280.0, "trend": "RISING", "trend_pct": 1.1, "distance_km": 48, "msp": 2275.0},
    {"commodity": "Wheat", "state": "Haryana", "district": "Karnal", "mandi": "Karnal Mandi", "modal_price": 2290.0, "min_price": 2260.0, "max_price": 2315.0, "trend": "RISING", "trend_pct": 3.1, "distance_km": 25, "msp": 2275.0},
    {"commodity": "Wheat", "state": "Uttar Pradesh", "district": "Agra", "mandi": "Agra Mandi", "modal_price": 2240.0, "min_price": 2210.0, "max_price": 2260.0, "trend": "FALLING", "trend_pct": -1.5, "distance_km": 20, "msp": 2275.0},
    {"commodity": "Wheat", "state": "Madhya Pradesh", "district": "Indore", "mandi": "Indore Malwa Mandi", "modal_price": 2310.0, "min_price": 2280.0, "max_price": 2340.0, "trend": "RISING", "trend_pct": 4.2, "distance_km": 30, "msp": 2275.0},

    # Paddy / Rice
    {"commodity": "Paddy", "state": "Punjab", "district": "Amritsar", "mandi": "Amritsar Grain Market", "modal_price": 2210.0, "min_price": 2180.0, "max_price": 2240.0, "trend": "RISING", "trend_pct": 1.8, "distance_km": 15, "msp": 2183.0},
    {"commodity": "Paddy", "state": "Haryana", "district": "Ambala", "mandi": "Ambala City Mandi", "modal_price": 2225.0, "min_price": 2190.0, "max_price": 2250.0, "trend": "RISING", "trend_pct": 2.2, "distance_km": 28, "msp": 2183.0},
    {"commodity": "Paddy", "state": "Uttar Pradesh", "district": "Gorakhpur", "mandi": "Gorakhpur Mandi", "modal_price": 2190.0, "min_price": 2160.0, "max_price": 2210.0, "trend": "STABLE", "trend_pct": 0.1, "distance_km": 18, "msp": 2183.0},

    # Cotton
    {"commodity": "Cotton", "state": "Punjab", "district": "Bathinda", "mandi": "Bathinda Cotton Market", "modal_price": 6780.0, "min_price": 6650.0, "max_price": 6900.0, "trend": "RISING", "trend_pct": 3.8, "distance_km": 22, "msp": 6620.0},
    {"commodity": "Cotton", "state": "Maharashtra", "district": "Nagpur", "mandi": "Nagpur Cotton Yard", "modal_price": 6820.0, "min_price": 6700.0, "max_price": 6950.0, "trend": "RISING", "trend_pct": 4.5, "distance_km": 40, "msp": 6620.0},

    # Mustard
    {"commodity": "Mustard", "state": "Rajasthan", "district": "Bharatpur", "mandi": "Bharatpur Mustard Mandi", "modal_price": 5780.0, "min_price": 5680.0, "max_price": 5850.0, "trend": "RISING", "trend_pct": 2.9, "distance_km": 18, "msp": 5650.0},
    {"commodity": "Mustard", "state": "Haryana", "district": "Hisar", "mandi": "Hisar Grain Market", "modal_price": 5740.0, "min_price": 5640.0, "max_price": 5810.0, "trend": "STABLE", "trend_pct": 0.5, "distance_km": 30, "msp": 5650.0},

    # Potato & Tomato
    {"commodity": "Potato", "state": "Uttar Pradesh", "district": "Farrukhabad", "mandi": "Farrukhabad Mandi", "modal_price": 1320.0, "min_price": 1280.0, "max_price": 1360.0, "trend": "RISING", "trend_pct": 5.2, "distance_km": 25, "msp": 1250.0},
    {"commodity": "Tomato", "state": "Maharashtra", "district": "Nashik", "mandi": "Nashik Agri Market", "modal_price": 1580.0, "min_price": 1450.0, "max_price": 1700.0, "trend": "RISING", "trend_pct": 6.8, "distance_km": 35, "msp": 1400.0},
]


class MandiRecommendRequest(BaseModel):
    commodity: str = Field(default="Wheat", description="Wheat | Paddy | Cotton | Mustard | Potato | Tomato")
    quantity_quintals: float = Field(gt=0, default=50.0, description="Harvested quantity in quintals")
    state: str = Field(default="Punjab", description="State name")
    farmer_district: Optional[str] = Field(default="Ludhiana", description="District name")


class RecommendedMandiItem(BaseModel):
    rank: int
    mandi: str
    district: str
    state: str
    modal_price_per_qtl: float
    distance_km: int
    trend: str
    trend_pct: float
    gross_revenue_inr: float
    estimated_transport_cost_inr: float
    net_revenue_inr: float
    net_extra_profit_vs_baseline_inr: float
    is_top_recommendation: bool


class MandiRecommendResponse(BaseModel):
    commodity: str
    quantity_quintals: float
    state: str
    farmer_district: str
    recommendations: List[RecommendedMandiItem]
    top_mandi_name: str
    max_net_revenue_inr: float
    total_extra_profit_inr: float
    recommendation_summary: str


class MandiPriceEngine:
    @staticmethod
    def get_rates(commodity: str = "Wheat", state: str = "Punjab") -> List[Dict[str, Any]]:
        """Filters mandi database by commodity and state."""
        filtered = [
            m for m in MANDI_DATABASE 
            if m["commodity"].lower() == commodity.lower() and m["state"].lower() == state.lower()
        ]
        if not filtered:
            # Fallback to commodity filter only
            filtered = [m for m in MANDI_DATABASE if m["commodity"].lower() == commodity.lower()]
        return filtered if filtered else MANDI_DATABASE[:3]

    @staticmethod
    def recommend_best_mandi(req: MandiRecommendRequest) -> MandiRecommendResponse:
        rates = MandiPriceEngine.get_rates(req.commodity, req.state)
        
        # Transport rate: Rs. 1.2 per quintal per km (approx Rs 12 per Qtl per 10 km)
        TRANSPORT_RATE_PER_QTL_KM = 1.20

        evaluated = []
        for idx, m in enumerate(rates):
            gross_rev = round(m["modal_price"] * req.quantity_quintals, 2)
            trans_cost = round(m["distance_km"] * TRANSPORT_RATE_PER_QTL_KM * req.quantity_quintals, 2)
            net_rev = round(gross_rev - trans_cost, 2)
            
            evaluated.append({
                "mandi_data": m,
                "gross_revenue": gross_rev,
                "transport_cost": trans_cost,
                "net_revenue": net_rev,
            })

        # Sort descending by net revenue
        evaluated.sort(key=lambda x: x["net_revenue"], reverse=True)
        baseline_net = evaluated[-1]["net_revenue"] if evaluated else 0.0

        recommendations = []
        for rank, item in enumerate(evaluated, start=1):
            m = item["mandi_data"]
            extra_profit = round(item["net_revenue"] - baseline_net, 2)
            
            recommendations.append(RecommendedMandiItem(
                rank=rank,
                mandi=m["mandi"],
                district=m["district"],
                state=m["state"],
                modal_price_per_qtl=m["modal_price"],
                distance_km=m["distance_km"],
                trend=m["trend"],
                trend_pct=m["trend_pct"],
                gross_revenue_inr=item["gross_revenue"],
                estimated_transport_cost_inr=item["transport_cost"],
                net_revenue_inr=item["net_revenue"],
                net_extra_profit_vs_baseline_inr=extra_profit,
                is_top_recommendation=(rank == 1)
            ))

        top_item = recommendations[0] if recommendations else None
        top_mandi = top_item.mandi if top_item else "Local Mandi"
        max_net = top_item.net_revenue_inr if top_item else 0.0
        extra_profit = top_item.net_extra_profit_vs_baseline_inr if top_item else 0.0

        summary = (
            f"Recommended Mandi: '{top_mandi}' ({top_item.distance_km} km away) offering ₹{top_item.modal_price_per_qtl}/Qtl. "
            f"After ₹{top_item.estimated_transport_cost_inr} transport cost, net revenue is ₹{max_net:,.2f} "
            f"(Extra Profit: +₹{extra_profit:,.2f} vs local market)."
        )

        return MandiRecommendResponse(
            commodity=req.commodity,
            quantity_quintals=req.quantity_quintals,
            state=req.state,
            farmer_district=req.farmer_district or "Local District",
            recommendations=recommendations,
            top_mandi_name=top_mandi,
            max_net_revenue_inr=max_net,
            total_extra_profit_inr=extra_profit,
            recommendation_summary=summary
        )
