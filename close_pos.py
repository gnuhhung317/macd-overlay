    def close_position(self, symbol: str, side: OrderSide = None, params={}) -> Order:
        """
        closes an open position for a market

        https://www.bitget.com/api-doc/contract/trade/Flash-Close-Position
        https://www.bitget.com/api-doc/uta/trade/Close-All-Positions

        :param str symbol: unified CCXT market symbol
        :param str [side]: one-way mode: 'buy' or 'sell', hedge-mode: 'long' or 'short'
        :param dict [params]: extra parameters specific to the exchange API endpoint
        :param boolean [params.uta]: set to True for the unified trading account(uta), defaults to False
        :returns dict: An `order structure <https://docs.ccxt.com/?id=order-structure>`
        """
        self.load_markets()
        market = self.market(symbol)
        request: dict = {
            'symbol': market['id'],
        }
        productType = None
        uta = None
        response = None
        productType, params = self.handle_product_type_and_params(market, params)
        uta, params = self.handle_option_and_params(params, 'closePosition', 'uta', False)
        if uta:
            if side is not None:
                request['posSide'] = side
            request['category'] = productType
            response = self.privateUtaPostV3TradeClosePositions(self.extend(request, params))
            #
            #     {
            #         "code": "00000",
            #         "msg": "success",
            #         "requestTime": 1751020218384,
            #         "data": {
            #             "list": [
            #                 {
            #                     "orderId": "1322440134099320832",
            #                     "clientOid": "1322440134099320833"
            #                 }
            #             ]
            #         }
            #     }
            #
        else:
            if side is not None:
                request['holdSide'] = side
            request['productType'] = productType
            response = self.privateMixPostV2MixOrderClosePositions(self.extend(request, params))
            #
            #     {
            #         "code": "00000",
            #         "msg": "success",
            #         "requestTime": 1702975017017,
            #         "data": {
            #             "successList": [
            #                 {
            #                     "orderId": "1120923953904893955",
            #                     "clientOid": "1120923953904893956"
            #                 }
            #             ],
            #             "failureList": [],
            #             "result": False
            #         }
            #     }
            #
        data = self.safe_value(response, 'data', {})
        order = self.safe_list_2(data, 'successList', 'list', [])
        return self.parse_order(order[0], market)
