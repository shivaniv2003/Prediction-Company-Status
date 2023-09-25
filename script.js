document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('status-form');
    const result = document.getElementById('result');
  
    form.addEventListener('submit', function (event) {
      event.preventDefault();
  
      // Retrieve user inputs
      const totalFunding = parseFloat(document.getElementById('total_f').value);
      const fundingRounds = parseInt(document.getElementById('funding_rounds').value);
      const seedFunding = parseFloat(document.getElementById('seed').value);
      const ventureFunding = parseFloat(document.getElementById('venture').value);
      const market = document.getElementById('market').value;
      const debtFinancing = parseFloat(document.getElementById('debt_financing').value);
      const countryCode = document.getElementById('country_code').value;
      const stateCode = document.getElementById('state_code').value;
  
      // Initialize company status to "Closed" by default
      let companyStatus = 'Closed';
  
      // Determine company status based on criteria
      if (totalFunding >= 1000000 && fundingRounds >= 3 && ventureFunding > 0) {
        companyStatus = 'Operating';
      } else if (totalFunding >= 500000 && fundingRounds >= 2 && ventureFunding > 0) {
        companyStatus = 'Acquired';
      }
  
      // Display the result
      result.textContent = `Company Status: ${companyStatus}`;
    });
  });
  

