import React, { useState } from "react";
import PropTypes from "prop-types";
import {
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow,
} from "@material-ui/core";
import styles from "./mytable.module.css";

// self-defined-components
const columns = [
  { id: "firm", label: "公司名稱", minWidth: 170 },
  {
    id: "investRatio",
    label: "投資比例",
    minWidth: 170,
    align: "right",
    format: (value) => value.toLocaleString("en-US"),
  },
  {
    id: "expectedProfit",
    label: "預期報酬",
    minWidth: 170,
    align: "right",
    format: (value) => value.toLocaleString("en-US"),
  },
];

function createData(firm, investRatio, expectedProfit) {
  investRatio = `${(investRatio * 100).toFixed(2)} %`;
  expectedProfit = expectedProfit - 1;
  return { firm, investRatio, expectedProfit };
}

const MyTable = ({ investData, totalProfit }) => {
  const [page, setPage] = useState(0);
  const rowsPerPage = 10;
  const rows = investData.map(({ firm, investRatio, expectedProfit }) => {
    return createData(firm, investRatio, expectedProfit);
  });

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  if (investData.length === 0) return <div></div>;
  return (
    <Paper className={styles.root}>
      <TableContainer className={styles.container}>
        <Table stickyHeader>
          <TableHead>
            <TableRow>
              {columns.map((column) => (
                <TableCell
                  key={column.id}
                  align={column.align}
                  style={{ minWidth: column.minWidth }}
                >
                  {column.label}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {rows
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((row) => {
                return (
                  <TableRow hover role="checkbox" tabIndex={-1} key={row.firm}>
                    {columns.map((column) => {
                      const value = row[column.id];
                      return (
                        <TableCell key={column.id} align={column.align}>
                          {column.format && typeof value === "number"
                            ? column.format(value)
                            : value}
                        </TableCell>
                      );
                    })}
                  </TableRow>
                );
              })}
            <TableRow>
              <TableCell rowSpan={2} />
              <TableCell>預期總報酬</TableCell>
              <TableCell align="right">{totalProfit}</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        component="div"
        count={rows.length}
        rowsPerPage={10}
        rowsPerPageOptions={[10]}
        page={page}
        onChangePage={handleChangePage}
      />
    </Paper>
  );
};

MyTable.propTypes = {
  investData: PropTypes.array.isRequired,
  totalProfit: PropTypes.number.isRequired,
};

export default MyTable;
