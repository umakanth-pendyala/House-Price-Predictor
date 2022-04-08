import { Component } from '@angular/core';
import { NgForm } from '@angular/forms';
import { RestService } from './rest.service';
import Swal from 'sweetalert2';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {


  constructor(private restServices: RestService) {
    
  }

  showSpinner: boolean = false;


  onSubmit(formData: NgForm) {

    this.showSpinner = true;
    
    this.restServices.fetchPredictions(formData.value).subscribe({
      next: (response: any) => {
        console.log(response)
        this.showSpinner = false;
        let htmlInSwal = '<div class="info-cstm">' + 
        '<p style="margin: 0">Decision Tree Predictions: ' + response.DTC + '</p>' +
        '<p style="margin: 0">Gaussian Naive Bayes Predictions: ' + response.GNB + '</p>' +
        '<p style="margin: 0">Simple Regression Predictions: ' + response.Simple_Regression + '</p>' +
        '<p style="margin: 0">Bagging Regression Predictions: ' + response.BGR + '</p>' +
        '<p style="margin: 0">Logestic Regression Predictions: ' + response.LGR + '</p>' +
      '</div>'
        Swal.fire({
          title: 'Predicted Values',
          icon: 'success',
          html: htmlInSwal,
          showCloseButton: true,
          confirmButtonText:
            '<i class="fa fa-thumbs-up"></i> Return',
          confirmButtonAriaLabel: 'Thumbs up, great!',
        })

      }, error: (e) => {
        console.log(e);
        Swal.fire({
          icon: 'error',
          title: 'Oops...',
          text: 'Something went wrong!',
        })
      }
    })
  }
}


