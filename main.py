from volume_analysis import volume_analysis

if __name__ == "__main__":
    min_buy_gold = 0
    max_buy_gold = 300
    min_sell_gold = 0
    max_sell_gold = 600
    min_vol_24h_gold = 1000
    max_vol_24h_gold = 0

    volume_analysis(
        min_buy_gold,
        max_buy_gold,
        min_sell_gold,
        max_sell_gold,
        min_vol_24h_gold,
        max_vol_24h_gold,
    )
