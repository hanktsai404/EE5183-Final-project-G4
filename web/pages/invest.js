import React, { useState } from "react";
import {
  FormControl,
  InputLabel,
  MenuItem,
  Typography,
  Select,
} from "@material-ui/core";
import styles from "../styles/test.module.css";

// components
import MyTable from "../components/MyTable";

const investDataSet = {
  low: {
    data: [
      { firm: "1326台化", investRatio: 0.0, expectedProfit: 0.992971698206425 },
      {
        firm: "2002中鋼",
        investRatio: 0.080747,
        expectedProfit: 1.00106948896945,
      },
      {
        firm: "2886兆豐金",
        investRatio: 0.0,
        expectedProfit: 1.00606853959516,
      },
      {
        firm: "2207和泰車",
        investRatio: 0.089143,
        expectedProfit: 1.01908279185134,
      },
      {
        firm: "2454聯發科",
        investRatio: 0.109905,
        expectedProfit: 1.00819659504984,
      },
      {
        firm: "2317鴻海",
        investRatio: 0.082547,
        expectedProfit: 1.00691154438322,
      },
      {
        firm: "3711日月光投控",
        investRatio: 0.077812,
        expectedProfit: 1.00703551921658,
      },
      { firm: "2891中信金", investRatio: 0.0, expectedProfit: 1.0027672797796 },
      {
        firm: "3008大立光",
        investRatio: 0.0,
        expectedProfit: 1.00279927281725,
      },
      {
        firm: "2330台積電",
        investRatio: 0.09997,
        expectedProfit: 1.00749375857284,
      },
      {
        firm: "2303聯電",
        investRatio: 0.124127,
        expectedProfit: 1.00520908653226,
      },
      {
        firm: "2881富邦金",
        investRatio: 0.079658,
        expectedProfit: 1.00534220677472,
      },
      {
        firm: "2308台達電",
        investRatio: 0.099091,
        expectedProfit: 1.00587134087544,
      },
      {
        firm: "6505台塑化",
        investRatio: 0.0,
        expectedProfit: 0.995885197892647,
      },
      {
        firm: "3045台灣大",
        investRatio: 0.0,
        expectedProfit: 1.00139613486596,
      },
      {
        firm: "2412中華電",
        investRatio: 0.078223,
        expectedProfit: 1.00445644231363,
      },
      { firm: "1303南亞", investRatio: 0.0, expectedProfit: 0.99880846134353 },
      { firm: "1216統一", investRatio: 0.0, expectedProfit: 1.00140334201776 },
      {
        firm: "2882國泰金",
        investRatio: 0.078778,
        expectedProfit: 1.00581313492966,
      },
      { firm: "1301台塑", investRatio: 0.0, expectedProfit: 0.998293525023066 },
    ],
    totalProfit: 0.007016906499310194,
  },
  medium: {
    data: [
      { firm: "1326台化", investRatio: 0.0, expectedProfit: 0.992971698206425 },
      {
        firm: "2002中鋼",
        investRatio: 0.012824,
        expectedProfit: 1.00106948896945,
      },
      {
        firm: "2886兆豐金",
        investRatio: 0.0,
        expectedProfit: 1.00606853959516,
      },
      {
        firm: "2207和泰車",
        investRatio: 0.076886,
        expectedProfit: 1.01908279185134,
      },
      {
        firm: "2454聯發科",
        investRatio: 0.232533,
        expectedProfit: 1.00819659504984,
      },
      {
        firm: "2317鴻海",
        investRatio: 0.026429,
        expectedProfit: 1.00691154438322,
      },
      {
        firm: "3711日月光投控",
        investRatio: 0.0,
        expectedProfit: 1.00703551921658,
      },
      { firm: "2891中信金", investRatio: 0.0, expectedProfit: 1.0027672797796 },
      {
        firm: "3008大立光",
        investRatio: 0.0,
        expectedProfit: 1.00279927281725,
      },
      {
        firm: "2330台積電",
        investRatio: 0.157211,
        expectedProfit: 1.00749375857284,
      },
      {
        firm: "2303聯電",
        investRatio: 0.338581,
        expectedProfit: 1.00520908653226,
      },
      {
        firm: "2881富邦金",
        investRatio: 0.004765,
        expectedProfit: 1.00534220677472,
      },
      {
        firm: "2308台達電",
        investRatio: 0.150772,
        expectedProfit: 1.00587134087544,
      },
      {
        firm: "6505台塑化",
        investRatio: 0.0,
        expectedProfit: 0.995885197892647,
      },
      {
        firm: "3045台灣大",
        investRatio: 0.0,
        expectedProfit: 1.00139613486596,
      },
      {
        firm: "2412中華電",
        investRatio: 0.0,
        expectedProfit: 1.00445644231363,
      },
      { firm: "1303南亞", investRatio: 0.0, expectedProfit: 0.99880846134353 },
      { firm: "1216統一", investRatio: 0.0, expectedProfit: 1.00140334201776 },
      {
        firm: "2882國泰金",
        investRatio: 0.0,
        expectedProfit: 1.00581313492966,
      },
      { firm: "1301台塑", investRatio: 0.0, expectedProfit: 0.998293525023066 },
    ],
    totalProfit: 0.007423047131985383,
  },
  high: {
    data: [
      { firm: "1326台化", investRatio: 0.0, expectedProfit: 0.992971698206425 },
      {
        firm: "2002中鋼",
        investRatio: 0.008812,
        expectedProfit: 1.00106948896945,
      },
      {
        firm: "2886兆豐金",
        investRatio: 0.0,
        expectedProfit: 1.00606853959516,
      },
      {
        firm: "2207和泰車",
        investRatio: 0.075169,
        expectedProfit: 1.01908279185134,
      },
      {
        firm: "2454聯發科",
        investRatio: 0.23638,
        expectedProfit: 1.00819659504984,
      },
      {
        firm: "2317鴻海",
        investRatio: 0.022903,
        expectedProfit: 1.00691154438322,
      },
      {
        firm: "3711日月光投控",
        investRatio: 0.0,
        expectedProfit: 1.00703551921658,
      },
      { firm: "2891中信金", investRatio: 0.0, expectedProfit: 1.0027672797796 },
      {
        firm: "3008大立光",
        investRatio: 0.0,
        expectedProfit: 1.00279927281725,
      },
      {
        firm: "2330台積電",
        investRatio: 0.158361,
        expectedProfit: 1.00749375857284,
      },
      {
        firm: "2303聯電",
        investRatio: 0.346217,
        expectedProfit: 1.00520908653226,
      },
      {
        firm: "2881富邦金",
        investRatio: 0.000465,
        expectedProfit: 1.00534220677472,
      },
      {
        firm: "2308台達電",
        investRatio: 0.151693,
        expectedProfit: 1.00587134087544,
      },
      {
        firm: "6505台塑化",
        investRatio: 0.0,
        expectedProfit: 0.995885197892647,
      },
      {
        firm: "3045台灣大",
        investRatio: 0.0,
        expectedProfit: 1.00139613486596,
      },
      {
        firm: "2412中華電",
        investRatio: 0.0,
        expectedProfit: 1.00445644231363,
      },
      { firm: "1303南亞", investRatio: 0.0, expectedProfit: 0.99880846134353 },
      { firm: "1216統一", investRatio: 0.0, expectedProfit: 1.00140334201776 },
      {
        firm: "2882國泰金",
        investRatio: 0.0,
        expectedProfit: 1.00581313492966,
      },
      { firm: "1301台塑", investRatio: 0.0, expectedProfit: 0.998293525023066 },
    ],
    totalProfit: 0.007422983807226835,
  },
};

// self-defined-components
const invest = () => {
  const [risk, setRisk] = useState("low");
  const [investData, setInvestData] = useState({ data: [], totalProfit: 0 });

  const handleOnChange = (e) => {
    setRisk(e.target.value);
    setInvestData(investDataSet[e.target.value]);
  };

  return (
    <div className={styles.root}>
      <div className={styles.displayBox}>
        <Typography className={styles.title} variant="h1">
          Financial Advisory Bot
        </Typography>
      </div>
      <div className={styles.displayBox}>
        <FormControl className={styles.formControl}>
          <InputLabel>請選擇你偏好的的風險程度</InputLabel>
          <Select
            id="demo-simple-select"
            labelId="demo-simple-select-label"
            value={risk}
            onChange={(e) => handleOnChange(e)}
          >
            <MenuItem value={"low"}>低</MenuItem>
            <MenuItem value={"medium"}>中</MenuItem>
            <MenuItem value={"high"}>高</MenuItem>
          </Select>
        </FormControl>
      </div>
      <div className={styles.displayBox}>
        <MyTable
          investData={investData["data"]}
          totalProfit={investData["totalProfit"]}
        />
      </div>
    </div>
  );
};

export default invest;
