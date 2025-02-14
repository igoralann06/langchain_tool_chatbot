<?php
header("Content-Type: application/json");

// Check if orderID is provided
if (!isset($_GET['orderID'])) {
    echo json_encode(["error" => "Missing orderID parameter"]);
    exit;
}

$orderID = $_GET['orderID'];

// Sample response data (Replace with real database queries)
$data = [
    "getOpenedOrders" => ["orderID" => $orderID, "result" => "getOpendOrders() API is called"],
    "getOrder" => ["orderID" => $orderID, "result" => "getOrder() API is called"],
    "getInvoiceNo" => ["orderID" => $orderID, "result" => "getInvoiceNo() API is called"],
    "getInvoiceDate" => ["orderID" => $orderID, "result" => "getInvoiceDate() API is called"],
    "isInvoiceIssued" => ["orderID" => $orderID, "result" => "isInvoiceIssued() API is called"],
    "getBillingInfo" => ["orderID" => $orderID, "result" => "getBillingInfo() API is called"],
    "getShippingInfo" => ["orderID" => $orderID, "result" => "getShippingInfo() API is called"],
    "getTotalPrice" => ["orderID" => $orderID, "result" => "getTotalPrice() API is called"],
    "getShippingStatus" => ["orderID" => $orderID, "result" => "getShippingStatus() API is called"],
    "getOrderStatus" => ["orderID" => $orderID, "result" => "getOrderStatus() API is called"],
    "getPaymentStatus" => ["orderID" => $orderID, "result" => "getPaymentStatus() API is called"],
    "getDetailedPriceInfo" => ["orderID" => $orderID, "result" => "getDetailedPriceInfo() API is called"],
    "getTotalWeight" => ["orderID" => $orderID, "result" => "getTotalWeight() API is called"]
];

// Get the requested endpoint
$endpoint = $_GET['endpoint'] ?? '';

if (array_key_exists($endpoint, $data)) {
    echo json_encode($data[$endpoint]);
} else {
    http_response_code(404);
    echo json_encode(["error" => "Invalid endpoint"]);
}
?>
