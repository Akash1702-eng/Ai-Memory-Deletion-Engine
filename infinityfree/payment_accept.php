<?php
/**
 * Razorpay Payment Bridge for AI Memory Deletion Engine
 * Host this file at: https://onlineclothier.infinityfreeapp.com/payment_accept.php
 */

// ── Razorpay Configuration ───────────────────────────────────────────────────
$razorpay_key_id = "rzp_live_SzziyLkZVWSPYh"; // Replace with your Razorpay Key ID

// Receive plan & user parameters
$user_id    = isset($_GET['user_id']) ? htmlspecialchars($_GET['user_id']) : '';
$user_email = isset($_GET['email']) ? htmlspecialchars($_GET['email']) : '';
$plan_id    = isset($_GET['plan_id']) ? htmlspecialchars($_GET['plan_id']) : 'plan_1m';
$return_url = isset($_GET['return_url']) ? $_GET['return_url'] : 'http://127.0.0.1:8001/#payment_success';

// Map plans to INR amounts (in paise: 1 INR = 100 paise)
$plans = [
    'plan_1m'  => ['name' => '1 Month Premium Subscription', 'amount' => 100,    'display' => '₹1'],
    'plan_6m'  => ['name' => '6 Months Premium Subscription', 'amount' => 249900, 'display' => '₹2,499'],
    'plan_12m' => ['name' => '12 Months Premium Subscription', 'amount' => 449900, 'display' => '₹4,499'],
];

$selected_plan = isset($plans[$plan_id]) ? $plans[$plan_id] : $plans['plan_1m'];
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Complete Checkout — AI Memory Deletion Engine</title>
    <style>
        body {
            background-color: #0b0f19;
            color: #f3f4f6;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
        }
        .card {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 16px;
            padding: 32px;
            width: 100%;
            max-width: 440px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
            text-align: center;
            box-sizing: border-box;
        }
        h2 { margin-top: 0; color: #8b5cf6; }
        .plan-box {
            background: #1f2937;
            border-radius: 12px;
            padding: 16px;
            margin: 20px 0;
        }
        .price { font-size: 32px; font-weight: bold; color: #10b981; margin: 8px 0; }
        .email-field {
            margin: 16px 0;
            text-align: left;
        }
        .email-field label {
            display: block;
            font-size: 13px;
            color: #9ca3af;
            margin-bottom: 6px;
        }
        .email-field input {
            width: 100%;
            padding: 12px;
            background: #0b0f19;
            border: 1px solid #374151;
            border-radius: 8px;
            color: #ffffff;
            font-size: 14px;
            box-sizing: border-box;
        }
        .btn-pay {
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
            color: white;
            border: none;
            padding: 14px 28px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            transition: transform 0.2s;
        }
        .btn-pay:hover { transform: translateY(-2px); }
    </style>
</head>
<body>
    <div class="card">
        <h2>⚡ MD Engine Premium</h2>
        <p>Complete your subscription payment to unlock unlimited AI Fine-Tuning & Gradient Ascent Machine Unlearning.</p>
        
        <div class="email-field">
            <label>Subscribing Account Email:</label>
            <input type="email" id="user-email-input" value="<?php echo $user_email; ?>" placeholder="Enter your registered account email" required />
        </div>

        <div class="plan-box">
            <h3><?php echo $selected_plan['name']; ?></h3>
            <div class="price"><?php echo $selected_plan['display']; ?></div>
            <p style="font-size: 13px; color: #9ca3af;">Instant Activation after payment</p>
        </div>

        <button id="rzp-button" class="btn-pay">Pay <?php echo $selected_plan['display']; ?> via Razorpay</button>
    </div>

    <!-- Razorpay Checkout Script -->
    <script src="https://checkout.razorpay.com/v1/checkout.js"></script>
    <script>
        document.getElementById('rzp-button').onclick = function(e){
            e.preventDefault();
            var emailVal = document.getElementById('user-email-input').value.trim();
            if (!emailVal) {
                alert('Please enter your account email address.');
                return;
            }

            var options = {
                "key": "<?php echo $razorpay_key_id; ?>",
                "amount": "<?php echo $selected_plan['amount']; ?>",
                "currency": "INR",
                "name": "AI Memory Deletion Engine",
                "description": "<?php echo $selected_plan['name']; ?>",
                "image": "https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
                "prefill": {
                    "email": emailVal
                },
                "handler": function (response){
                    var returnUrl = "<?php echo $return_url; ?>";
                    var delimiter = returnUrl.indexOf('?') !== -1 ? '&' : '?';
                    var redirectUrl = returnUrl + delimiter +
                        "payment_id=" + encodeURIComponent(response.razorpay_payment_id) +
                        "&plan_id=<?php echo $plan_id; ?>" +
                        "&user_id=<?php echo $user_id; ?>" +
                        "&email=" + encodeURIComponent(emailVal);
                    window.location.href = redirectUrl;
                },
                "theme": {
                    "color": "#8b5cf6"
                }
            };
            var rzp = new Razorpay(options);
            rzp.open();
        };

        // Auto trigger Razorpay if email is prefilled
        window.onload = function() {
            var emailVal = document.getElementById('user-email-input').value.trim();
            if (emailVal) {
                document.getElementById('rzp-button').click();
            }
        };
    </script>
</body>
</html>
