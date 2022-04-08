import { HttpClient } from "@angular/common/http";
import { Injectable } from "@angular/core";


@Injectable()
export class RestService {
    constructor(private http: HttpClient) {}


    fetchPredictions(parameters: any) {
        return this.http.post('http://localhost:8000/postHouseData', parameters);
    }
}