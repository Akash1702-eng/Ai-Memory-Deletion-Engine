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
    'plan_1m'  => [
        'name'        => '1 Month Premium Plan',
        'tag'         => 'Monthly Subscription',
        'amount'      => 100,
        'display'     => '₹1',
        'period'      => '/month',
        'duration'    => '30 Days Access'
    ],
    'plan_6m'  => [
        'name'        => '6 Months Premium Plan',
        'tag'         => 'Semi-Annual (Save 15%)',
        'amount'      => 249900,
        'display'     => '₹2,499',
        'period'      => '/6 months',
        'duration'    => '180 Days Access'
    ],
    'plan_12m' => [
        'name'        => '12 Months Premium Plan',
        'tag'         => 'Annual Best Value (Save 30%)',
        'amount'      => 449900,
        'display'     => '₹4,499',
        'period'      => '/year',
        'duration'    => '365 Days Access'
    ],
];

$selected_plan = isset($plans[$plan_id]) ? $plans[$plan_id] : $plans['plan_1m'];
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Secure Checkout — AI Memory Deletion Engine</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: #f8fafc;
            background-image: 
                radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 90%, rgba(139, 92, 246, 0.05) 0%, transparent 40%);
            color: #0f172a;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            padding: 24px 16px;
            -webkit-font-smoothing: antialiased;
        }

        .checkout-wrapper {
            width: 100%;
            max-width: 460px;
        }

        .checkout-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 36px 32px;
            box-shadow: 
                0 20px 48px -10px rgba(15, 23, 42, 0.08),
                0 8px 18px -4px rgba(15, 23, 42, 0.04);
            position: relative;
            overflow: hidden;
        }

        .checkout-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
        }

        /* ── Header ────────────────────────────────────────── */
        .brand-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
        }

        .brand-icon {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
            flex-shrink: 0;
        }

        .brand-text h1 {
            font-size: 17px;
            font-weight: 700;
            color: #0f172a;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }

        .brand-text p {
            font-size: 12px;
            color: #64748b;
            font-weight: 500;
        }

        /* ── Plan Summary Box ──────────────────────────────── */
        .plan-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 24px;
        }

        .plan-tag {
            display: inline-block;
            background: #eef2ff;
            color: #4338ca;
            font-size: 11px;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 9999px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .plan-title-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 14px;
        }

        .plan-name {
            font-size: 16px;
            font-weight: 700;
            color: #0f172a;
        }

        .plan-price {
            font-size: 26px;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.02em;
        }

        .plan-period {
            font-size: 13px;
            font-weight: 500;
            color: #64748b;
        }

        .plan-features {
            list-style: none;
            padding-top: 14px;
            border-top: 1px solid #e2e8f0;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .plan-feature-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: #334155;
            font-weight: 500;
        }

        .feature-check {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: rgba(5, 150, 105, 0.1);
            color: #059669;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        /* ── Input Group ───────────────────────────────────── */
        .form-group {
            margin-bottom: 24px;
            text-align: left;
        }

        .form-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            font-weight: 600;
            color: #334155;
            margin-bottom: 8px;
        }

        .form-input-wrap {
            position: relative;
        }

        .input-icon {
            position: absolute;
            left: 14px;
            top: 50%;
            transform: translateY(-50%);
            color: #94a3b8;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .form-input {
            width: 100%;
            padding: 13px 14px 13px 42px;
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            color: #0f172a;
            font-family: inherit;
            font-size: 14px;
            outline: none;
            transition: all 0.15s ease;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        }

        .form-input:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
        }

        /* ── Pay Button ────────────────────────────────────── */
        .btn-pay {
            width: 100%;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: #ffffff;
            border: none;
            padding: 15px 24px;
            font-size: 15px;
            font-weight: 700;
            font-family: inherit;
            border-radius: 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35);
            transition: all 0.2s ease;
        }

        .btn-pay:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.45);
        }

        .btn-pay:active {
            transform: translateY(0);
        }

        /* ── Security Trust Footer ─────────────────────────── */
        .security-footer {
            margin-top: 24px;
            padding-top: 18px;
            border-top: 1px solid #f1f5f9;
            text-align: center;
        }

        .security-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: #64748b;
            font-weight: 500;
            margin-bottom: 12px;
        }

        .payment-methods {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 14px;
            font-size: 11px;
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .method-item {
            display: flex;
            align-items: center;
            gap: 4px;
        }

        @media (max-width: 480px) {
            .checkout-card {
                padding: 28px 20px;
            }
            .plan-price {
                font-size: 22px;
            }
        }
    </style>
</head>
<body>

    <div class="checkout-wrapper">
        <div class="checkout-card">
            
            <!-- Brand Header -->
            <div class="brand-header">
                <div class="brand-icon">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a4 4 0 0 0-4 4v1a3 3 0 0 0-3 3 4 4 0 0 0 2 3.5 3 3 0 0 0-1 2.5 4 4 0 0 0 4 4h4a4 4 0 0 0 4-4 3 3 0 0 0-1-2.5 4 4 0 0 0 2-3.5 3 3 0 0 0-3-3V6a4 4 0 0 0-4-4z"/><path d="M12 2v20"/><path d="M4 12h16"/></svg>
                </div>
                <div class="brand-text">
                    <h1>AI Memory Engine</h1>
                    <p>Secure Subscription Checkout</p>
                </div>
            </div>

            <!-- Plan Summary Card -->
            <div class="plan-box">
                <div class="plan-tag"><?php echo $selected_plan['tag']; ?></div>
                <div class="plan-title-row">
                    <div class="plan-name"><?php echo $selected_plan['name']; ?></div>
                    <div class="plan-price"><?php echo $selected_plan['display']; ?><span class="plan-period"><?php echo $selected_plan['period']; ?></span></div>
                </div>
                <ul class="plan-features">
                    <li class="plan-feature-item">
                        <span class="feature-check">
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                        </span>
                        <span>Full LoRA Fine-Tuning & Gradient Ascent Access</span>
                    </li>
                    <li class="plan-feature-item">
                        <span class="feature-check">
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                        </span>
                        <span>3-Stage Verification & MIA Security Audit</span>
                    </li>
                    <li class="plan-feature-item">
                        <span class="feature-check">
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                        </span>
                        <span><?php echo $selected_plan['duration']; ?> with Instant Activation</span>
                    </li>
                </ul>
            </div>

            <!-- Email Form -->
            <div class="form-group">
                <label class="form-label" for="user-email-input">
                    <span>Account Email</span>
                    <span style="font-size: 11px; color: #64748b; font-weight: 500;">Linked to your Engine</span>
                </label>
                <div class="form-input-wrap">
                    <div class="input-icon">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>
                    </div>
                    <input 
                        type="email" 
                        id="user-email-input" 
                        class="form-input" 
                        value="<?php echo $user_email; ?>" 
                        placeholder="Enter registered account email" 
                        required 
                    />
                </div>
            </div>

            <!-- Pay CTA Button -->
            <button id="rzp-button" class="btn-pay">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
                <span>Pay <?php echo $selected_plan['display']; ?> with Razorpay</span>
            </button>

            <!-- Trust & Security Details -->
            <div class="security-footer">
                <div class="security-badge">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                    <span>256-bit SSL Encrypted & Secured by Razorpay</span>
                </div>
                <div class="payment-methods">
                    <span class="method-item">UPI</span>
                    <span>·</span>
                    <span class="method-item">GPay</span>
                    <span>·</span>
                    <span class="method-item">PhonePe</span>
                    <span>·</span>
                    <span class="method-item">Cards</span>
                    <span>·</span>
                    <span class="method-item">NetBanking</span>
                </div>
            </div>

        </div>
    </div>

    <!-- Razorpay Checkout Script -->
    <script src="https://checkout.razorpay.com/v1/checkout.js"></script>
    <script>
        document.getElementById('rzp-button').onclick = function(e){
            e.preventDefault();
            var emailVal = document.getElementById('user-email-input').value.trim();
            if (!emailVal) {
                alert('Please enter your account email address.');
                document.getElementById('user-email-input').focus();
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
                    "color": "#6366f1"
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
