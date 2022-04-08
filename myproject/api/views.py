from urllib import request
from rest_framework.response import Response
from rest_framework.decorators import api_view
import Price_Predictor as pp
import json


@api_view(['POST'])
def getPredictions(request):
    inputs = json.loads(request.body)
    bedrooms = inputs.get("bedrooms")
    sqftliving = inputs.get("sqftliving")
    condition = inputs.get("condition")
    grade = inputs.get("grade")
    prediction1 = pp.getDTC(bedrooms, sqftliving, condition, grade)[0]
    prediction2 = pp.getGNB(bedrooms, sqftliving, condition, grade)[0]
    prediction3 = pp.getRGRp(bedrooms, sqftliving, condition, grade)[0]
    prediction4 = pp.getBaggingRegression(bedrooms, sqftliving, condition, grade)[0]
    prediction5 = pp.getLogisticRegression(bedrooms, sqftliving, condition, grade)[0]
    return Response({ 'DTC': prediction1, 'GNB': prediction2, 'Simple_Regression': prediction3, 'BGR': prediction4, 'LGR': prediction5} )
