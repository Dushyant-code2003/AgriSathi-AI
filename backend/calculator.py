"""
AgriSathi Smart Fertilizer & Yield ROI Calculator Engine
Calculates land-specific NPK dosages, bag requirements, subsidized input costs, projected yield, and MSP revenue.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field

# Unit conversion to Standard Acres
UNIT_CONVERSION = {
    "acre": 1.0,
    "hectare": 2.47105,
    "bigha": 0.625,  # Standard Northern Bigha (1 Acre = 1.6 Bigha)
}

# Subsidized Fertilizer Prices in India (2024 Rates)
FERTILIZER_PRICES = {
    "urea": {"name": "Urea (46% N)", "bag_weight_kg": 45, "bag_price_inr": 266.50, "cost_per_kg": 5.92},
    "dap": {"name": "DAP (18:46:0)", "bag_weight_kg": 50, "bag_price_inr": 1350.00, "cost_per_kg": 27.00},
    "mop": {"name": "MOP (60% K2O)", "bag_weight_kg": 50, "bag_price_inr": 1700.00, "cost_per_kg": 34.00},
    "zinc": {"name": "Zinc Sulphate (21% Zn)", "bag_weight_kg": 10, "bag_price_inr": 600.00, "cost_per_kg": 60.00},
}

# Crop Agricultural Benchmarks (Per Acre)
CROP_DATA = {
    "Wheat": {
        "dap_kg_per_acre": 50.0,
        "urea_kg_per_acre": 60.0,
        "mop_kg_per_acre": 20.0,
        "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 18.5,
        "msp_per_quintal_inr": 2275.0, # Government MSP 2023-24
    },
    "Paddy": {
        "dap_kg_per_acre": 40.0,
        "urea_kg_per_acre": 75.0,
        "mop_kg_per_acre": 25.0,
        "zinc_kg_per_acre": 12.0,
        "avg_yield_quintal_per_acre": 22.0,
        "msp_per_quintal_inr": 2183.0,
    },
    "Cotton": {
        "dap_kg_per_acre": 45.0,
        "urea_kg_per_acre": 90.0,
        "mop_kg_per_acre": 30.0,
        "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 10.5,
        "msp_per_quintal_inr": 6620.0,
    },
    "Mustard": {
        "dap_kg_per_acre": 35.0,
        "urea_kg_per_acre": 45.0,
        "mop_kg_per_acre": 15.0,
        "zinc_kg_per_acre": 8.0,
        "avg_yield_quintal_per_acre": 8.5,
        "msp_per_quintal_inr": 5650.0,
    },
    "Sugarcane": {
        "dap_kg_per_acre": 100.0,
        "urea_kg_per_acre": 150.0,
        "mop_kg_per_acre": 60.0,
        "zinc_kg_per_acre": 15.0,
        "avg_yield_quintal_per_acre": 350.0,
        "msp_per_quintal_inr": 315.0, # Fair & Remunerative Price (FRP)
    },
    "Potato": {
        "dap_kg_per_acre": 80.0,
        "urea_kg_per_acre": 100.0,
        "mop_kg_per_acre": 50.0,
        "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 120.0,
        "msp_per_quintal_inr": 1250.0,
    },
    "Tomato": {
        "dap_kg_per_acre": 60.0,
        "urea_kg_per_acre": 80.0,
        "mop_kg_per_acre": 40.0,
        "zinc_kg_per_acre": 10.0,
        "avg_yield_quintal_per_acre": 110.0,
        "msp_per_quintal_inr": 1400.0,
    }
}

# Soil Adjustments Multipliers
SOIL_MULTIPLIERS = {
    "Alluvial": {"dap": 1.0, "urea": 1.0, "mop": 1.0, "zinc": 1.0},
    "Black":    {"dap": 0.9, "urea": 1.0, "mop": 0.85, "zinc": 1.1},
    "Clay":     {"dap": 1.05, "urea": 1.1, "mop": 0.95, "zinc": 1.0},
    "Sandy":    {"dap": 1.15, "urea": 1.2, "mop": 1.15, "zinc": 1.25},
    "Loamy":    {"dap": 0.95, "urea": 0.95, "mop": 0.95, "zinc": 0.95},
}


class CalculatorRequest(BaseModel):
    land_size: float = Field(gt=0, default=2.5, description="Land area")
    unit: str = Field(default="acre", description="acre | hectare | bigha")
    crop: str = Field(default="Wheat", description="Wheat | Paddy | Cotton | Mustard | Sugarcane | Potato | Tomato")
    soil_type: str = Field(default="Alluvial", description="Alluvial | Black | Clay | Sandy | Loamy")


class FertilizerBagDetail(BaseModel):
    name: str
    kg_required: float
    bags_required: float
    bag_weight_kg: float
    unit_price_inr: float
    total_cost_inr: float


class CalculatorResponse(BaseModel):
    land_size_input: float
    land_unit_input: str
    equivalent_acres: float
    crop: str
    soil_type: str
    fertilizers: List[FertilizerBagDetail]
    total_fertilizer_cost_inr: float
    estimated_yield_quintals: float
    estimated_gross_revenue_inr: float
    estimated_net_profit_inr: float
    roi_percentage: float


class AgriCalculatorEngine:
    @staticmethod
    def calculate(req: CalculatorRequest) -> CalculatorResponse:
        unit_key = req.unit.lower()
        multiplier = UNIT_CONVERSION.get(unit_key, 1.0)
        acres = round(req.land_size * multiplier, 2)

        crop_info = CROP_DATA.get(req.crop, CROP_DATA["Wheat"])
        soil_mult = SOIL_MULTIPLIERS.get(req.soil_type, SOIL_MULTIPLIERS["Alluvial"])

        # Calculate exact requirements
        dap_kg  = round(crop_info["dap_kg_per_acre"] * acres * soil_mult["dap"], 1)
        urea_kg = round(crop_info["urea_kg_per_acre"] * acres * soil_mult["urea"], 1)
        mop_kg  = round(crop_info["mop_kg_per_acre"] * acres * soil_mult["mop"], 1)
        zinc_kg = round(crop_info["zinc_kg_per_acre"] * acres * soil_mult["zinc"], 1)

        fert_configs = [
            ("dap", dap_kg),
            ("urea", urea_kg),
            ("mop", mop_kg),
            ("zinc", zinc_kg),
        ]

        fertilizer_details = []
        total_fert_cost = 0.0

        for key, kg in fert_configs:
            info = FERTILIZER_PRICES[key]
            bags = round(kg / info["bag_weight_kg"], 1)
            cost = round(kg * info["cost_per_kg"], 2)
            total_fert_cost += cost

            fertilizer_details.append(FertilizerBagDetail(
                name=info["name"],
                kg_required=kg,
                bags_required=bags,
                bag_weight_kg=info["bag_weight_kg"],
                unit_price_inr=info["bag_price_inr"],
                total_cost_inr=cost
            ))

        total_fert_cost = round(total_fert_cost, 2)

        # Economic Yield & MSP Revenue calculations
        yield_quintals = round(crop_info["avg_yield_quintal_per_acre"] * acres, 1)
        gross_revenue  = round(yield_quintals * crop_info["msp_per_quintal_inr"], 2)
        
        # Estimate total cultivation cost (fertilizer + labor/seeds/irrigation ~ 2.5x fert cost)
        total_estimated_cost = round(total_fert_cost * 2.5, 2)
        net_profit = round(gross_revenue - total_estimated_cost, 2)
        roi_pct = round((net_profit / max(1, total_estimated_cost)) * 100, 1)

        return CalculatorResponse(
            land_size_input=req.land_size,
            land_unit_input=req.unit,
            equivalent_acres=acres,
            crop=req.crop,
            soil_type=req.soil_type,
            fertilizers=fertilizer_details,
            total_fertilizer_cost_inr=total_fert_cost,
            estimated_yield_quintals=yield_quintals,
            estimated_gross_revenue_inr=gross_revenue,
            estimated_net_profit_inr=net_profit,
            roi_percentage=roi_pct
        )
